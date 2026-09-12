#!/usr/bin/env python3
"""Validate an FBPUNet checkpoint and write a rich prediction report.

Loads ``best*.ckpt`` + ``model_config.yaml`` (or a Lightning checkpoint that
embeds hyperparams), runs the fold/site validation set, and writes:

    report/eval_<label>_<YYYYMMDD_HHMMSS>/
      report.md  index.html  stats.html  worst.html  typical.html
      metrics.json  traces.parquet (or traces.csv.gz)
      figs/worst_*.png  figs/typical_*.png  figs/high_rmse_*.png

Example::

    python train/fbp_eval.py --ckpt-dir /home/mika/data/seismic_activity/weights/baseline/foldA \\
        --fold A --backend hdf5 --data-dir /tmp/data/
    python train/fbp_eval.py --ckpt output/train_foldA_resnet34/best-epoch=013-step=015232.ckpt \\
        --fold A --backend npz
    python train/fbp_eval.py --picker before_after --ckpt-dir output/train_foldA_resnet34-before-after \\
        --fold A --backend hdf5 --rmse-above 7
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(TRAIN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from seismic_utils.dataset import DEFAULT_DATA_DIR
from seismic_utils.fbp_eval_report import (
    annotate_trace_frame,
    gather_summary,
    headline_metrics,
    offset_bin_table,
    pick_gallery_gathers,
    pick_high_rmse_gathers,
    plot_gather_residual,
    write_index_html,
    write_report_md,
    write_stats_html,
    write_trace_table,
    write_worst_html,
)
from seismic_utils.fb_smooth import DEFAULT_SMOOTH_THRESHOLD
from seismic_utils.hardpicks_bridge import hardpicks_available, hardpicks_item_to_shot_gather
from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat
from seismic_utils.pickers import (
    PICKER_BEFORE_AFTER,
    PICKER_FBPUNET,
    make_eval_evaluator,
    picker_from_hparams,
    reconcile_cli_picker,
    spec_for,
)
from seismic_utils.predict import load_fbp_model, resolve_checkpoint

import fbp_train as train_cli

logger = logging.getLogger("fbp_eval")

EVAL_METRICS: List[Dict[str, Any]] = [
    {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 1}},
    {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 3}},
    {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 5}},
    {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 7}},
    {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 9}},
    {"metric_type": "MeanBiasError"},
    {"metric_type": "MeanAbsoluteError"},
    {"metric_type": "RootMeanSquaredError"},
    {"metric_type": "GatherCoverage"},
]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate an FBPUNet checkpoint and write a prediction report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--picker",
        choices=(PICKER_FBPUNET, PICKER_BEFORE_AFTER),
        default=None,
        help=(
            "Decode head. Default: checkpoint hparams (picker / segm_class_count). "
            "On mismatch the checkpoint wins."
        ),
    )
    p.add_argument("--ckpt", type=Path, default=None, help="Path to a .ckpt file.")
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory containing best*.ckpt and optional model_config.yaml.",
    )
    p.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="YAML config (default: <ckpt-dir>/model_config.yaml if present).",
    )
    p.add_argument("--fold", default=None, metavar="ID", help="Hardpicks fold (A–K). Uses the fold's valid sites.")
    p.add_argument(
        "--sites",
        default=None,
        help="Comma-separated eval sites (mutually exclusive with --fold).",
    )
    p.add_argument("--backend", choices=("npz", "hdf5"), default="npz")
    p.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    p.add_argument("--npz-root", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Defaults to the saved training batch size (legacy checkpoints: 4).")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--eval-ratio",
        type=float,
        default=None,
        help="If set with --sites, take this intra-site holdout (same splitter as training).",
    )
    p.add_argument("--n-worst", type=int, default=8, help="Worst gathers to plot.")
    p.add_argument("--n-typical", type=int, default=4, help="Typical/good gathers to plot.")
    p.add_argument(
        "--rmse-above",
        type=float,
        default=None,
        metavar="SAMPLES",
        help=(
            "Plot every gather whose RMSE (samples) is strictly greater than this "
            "value, overlaying reference and prediction. Omit to skip."
        ),
    )
    p.add_argument("--report-dir", type=Path, default=None, help="Report root (default: <repo>/report).")
    p.add_argument(
        "--smooth-threshold",
        type=int,
        default=None,
        help=(
            "Before/after pick smoother window in samples "
            f"(default: checkpoint hparams or {DEFAULT_SMOOTH_THRESHOLD})."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--list-folds", action="store_true")
    args = p.parse_args(argv)
    if not args.list_folds and args.ckpt is None and args.ckpt_dir is None:
        p.error("Provide --ckpt or --ckpt-dir")
    if args.rmse_above is not None and not np.isfinite(args.rmse_above):
        p.error("--rmse-above must be a finite number")
    return args


def resolve_eval_sites(args: argparse.Namespace) -> tuple[str, List[str], Optional[float]]:
    if args.fold and args.sites:
        raise SystemExit("Use either --fold or --sites, not both.")
    if args.fold:
        fold_id, _train, valid_sites = train_cli.resolve_fold(args.fold)
        return f"fold{fold_id}", valid_sites, None
    if args.sites:
        sites = [s.strip() for s in args.sites.split(",") if s.strip()]
        if not sites:
            raise SystemExit("--sites must list at least one site")
        label = "_".join(s.lower() for s in sites)
        return label, sites, args.eval_ratio
    raise SystemExit("Provide --fold or --sites for the evaluation split.")


def find_model_config(ckpt: Path, ckpt_dir: Optional[Path], explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"model config not found: {path}")
        return path
    candidates = []
    if ckpt_dir is not None:
        candidates.append(Path(ckpt_dir) / "model_config.yaml")
    candidates.append(ckpt.parent / "model_config.yaml")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def merge_eval_metrics(existing: Any) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()

    def _key(item: Dict[str, Any]) -> str:
        mtype = str(item.get("metric_type", ""))
        if mtype == "HitRate":
            buf = (item.get("metric_params") or {}).get("buffer_size_px")
            return f"HitRate{buf}px"
        return mtype

    def _as_metric_dict(raw: Any) -> Optional[Dict[str, Any]]:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            if not hasattr(raw, "keys"):
                return None
            raw = dict(raw)
        item = dict(raw)
        params = item.get("metric_params")
        if params is not None and not isinstance(params, dict) and hasattr(params, "keys"):
            item["metric_params"] = dict(params)
        return item

    for raw in list(existing or []) + EVAL_METRICS:
        item = _as_metric_dict(raw)
        if not item or "metric_type" not in item:
            continue
        key = _key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def sample_rate_ms_from_item(item: Dict[str, Any]) -> float:
    if "sample_rate_ms" in item:
        return float(item["sample_rate_ms"])
    meta = item.get("meta") or {}
    if "sample_rate_ms" in meta:
        return float(meta["sample_rate_ms"])
    return 2.0


def lookup_origin_rate(origin: str, rates: Dict[str, float], default: float = 2.0) -> float:
    if origin in rates:
        return float(rates[origin])
    origin_l = origin.lower()
    for key, value in rates.items():
        key_l = str(key).lower()
        if key_l in origin_l or origin_l in key_l:
            return float(value)
    return float(default)


def collect_sample_rates(parser, site_names: Sequence[str]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    wanted = {s.lower() for s in site_names}
    for i in range(len(parser)):
        meta = parser.get_meta_gather(i)
        origin = str(meta.get("origin") or meta.get("site_name") or "")
        if not origin or origin in rates:
            continue
        rates[origin] = sample_rate_ms_from_item(parser[i])
        have = {o.lower() for o in rates}
        if wanted and all(any(site in origin or origin in site for origin in have) for site in wanted):
            break
    return rates or {"unknown": 2.0}


def build_gather_index(parser) -> Dict[tuple, int]:
    index: Dict[tuple, int] = {}
    for i in range(len(parser)):
        meta = parser.get_meta_gather(i)
        origin = str(meta.get("origin") or meta.get("site_name") or "")
        key = (origin, int(meta["gather_id"]), int(meta["shot_id"]))
        index.setdefault(key, i)
    return index


def lookup_parser_index(
    gather_index: Dict[tuple, int],
    origin: str,
    gather_id: int,
    shot_id: int,
) -> Optional[int]:
    key = (origin, gather_id, shot_id)
    if key in gather_index:
        return gather_index[key]
    origin_l = origin.lower()
    for (cand_origin, cand_gather, cand_shot), idx in gather_index.items():
        if cand_gather != gather_id or cand_shot != shot_id:
            continue
        cand_l = str(cand_origin).lower()
        if cand_l in origin_l or origin_l in cand_l:
            return idx
    return None


def pred_ms_for_gather(item: Dict[str, Any], gdf: pd.DataFrame, dt_ms: float) -> np.ndarray:
    rec_ids = np.asarray(item["rec_ids"]).reshape(-1)
    pred_by_rec = {
        int(row.ReceiverId): float(row.Predictions)
        for row in gdf.itertuples(index=False)
        if pd.notna(row.Predictions)
    }
    idx = np.array([pred_by_rec.get(int(r), np.nan) for r in rec_ids], dtype=np.float64)
    pred_ms = idx * float(dt_ms)
    pred_ms[~np.isfinite(idx) | (idx <= 0)] = np.nan
    return pred_ms


def run_eval(model, loader, device: torch.device, evaluator) -> tuple[pd.DataFrame, Dict[str, float], float]:
    from seismic_utils.validation import mean_epoch_loss
    model.eval()
    evaluator.reset()
    losses = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="eval")):
            batch = batch_to_device(batch, device)
            _preds, loss, _metrics = model._generic_step(batch, batch_idx, evaluator)
            losses.append((float(loss.detach().cpu()), model._last_eval_loss_weight))
    evaluator.finalize()
    summary = evaluator.summarize()
    mean_loss = mean_epoch_loss(losses)
    return evaluator._dataframe.copy(), summary, mean_loss


def plot_gallery(
    rows: pd.DataFrame,
    *,
    parser,
    gather_index: Dict[tuple, int],
    traces: pd.DataFrame,
    figs_dir: Path,
    prefix: str,
    dt_by_origin: Dict[str, float],
) -> List[Dict[str, Any]]:
    figs_dir.mkdir(parents=True, exist_ok=True)
    cards: List[Dict[str, Any]] = []
    for i, row in enumerate(rows.itertuples(index=False), start=1):
        origin = str(getattr(row, "Origin", ""))
        parser_idx = lookup_parser_index(
            gather_index, origin, int(row.GatherId), int(row.ShotId)
        )
        if parser_idx is None:
            logger.warning(
                "gather not found in parser: %s g%s s%s",
                origin,
                row.GatherId,
                row.ShotId,
            )
            continue
        item = parser[parser_idx]
        gather = hardpicks_item_to_shot_gather(item)
        gdf = traces[
            (traces["OriginId"] == row.OriginId)
            & (traces["GatherId"] == row.GatherId)
            & (traces["ShotId"] == row.ShotId)
        ]
        dt = float(
            dt_by_origin.get(origin)
            or lookup_origin_rate(origin, dt_by_origin, gdf["SampleRateMs"].iloc[0] if len(gdf) else 2.0)
        )
        pred_ms = pred_ms_for_gather(item, gdf, dt)
        name = f"{prefix}_{i:02d}_g{int(row.GatherId)}_s{int(row.ShotId)}.png"
        rel = f"figs/{name}"
        subtitle = (
            f"{origin}  gather={int(row.GatherId)} shot={int(row.ShotId)} "
            f"RMSE={getattr(row, 'RMSE', float('nan')):.3g} "
            f"MAE={getattr(row, 'MAE', float('nan')):.3g} "
            f"HR@1={getattr(row, 'HitRate1px', float('nan')):.3f} "
            f"P90={getattr(row, 'P90AbsError', float('nan')):.3g}"
        )
        plot_gather_residual(gather, pred_ms, figs_dir / name, subtitle=subtitle)
        cards.append(
            {
                "image": rel,
                "label": f"{origin} g{int(row.GatherId)} shot={int(row.ShotId)}",
                "stats": (
                    f"RMSE={getattr(row, 'RMSE', float('nan')):.4g} samples  "
                    f"MAE={getattr(row, 'MAE', float('nan')):.4g} samples  "
                    f"HR@1={getattr(row, 'HitRate1px', float('nan')):.4f}  "
                    f"n_labeled={int(getattr(row, 'n_labeled', 0))}"
                ),
            }
        )
    return cards


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_folds:
        train_cli.list_folds()
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    print("PL compat:", ensure_hardpicks_lightning_compat())
    if not hardpicks_available():
        raise SystemExit("hardpicks (+ torch) required — run setup_lightning.sh / install requirements")

    try:
        ckpt = resolve_checkpoint(args.ckpt, ckpt_dir=args.ckpt_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    config_path = find_model_config(ckpt, args.ckpt_dir, args.model_config)
    file_cfg: Dict[str, Any] = {}
    if config_path is not None:
        loaded = yaml.safe_load(config_path.read_text()) or {}
        if isinstance(loaded, dict):
            file_cfg = loaded

    site_label, site_names, eval_ratio = resolve_eval_sites(args)
    npz_root = args.npz_root or (args.data_dir / "npz")

    import hardpicks.data.fbp.data_module as fbp_data_module
    import torch.utils.data

    model = load_fbp_model(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.images_to_display = 0
    hp = dict(getattr(model, "hparams", {}) or {})
    if file_cfg:
        hp = {**file_cfg, **hp}
    saved_batch_size = (hp.get("training_data") or {}).get("batch_size")
    if args.batch_size is None:
        args.batch_size = int(saved_batch_size or 4)
    elif saved_batch_size and args.batch_size != int(saved_batch_size):
        logger.warning("Batch size differs from training; batch-dependent padding can change predictions")
    hp["eval_metrics"] = merge_eval_metrics(hp.get("eval_metrics"))
    ckpt_picker = picker_from_hparams(hp)
    cli_picker = args.picker or ckpt_picker
    resolved_picker, mismatched = reconcile_cli_picker(cli_picker, ckpt_picker)
    if mismatched and args.picker:
        logger.warning(
            "checkpoint picker=%s disagrees with --picker %s; using checkpoint",
            ckpt_picker,
            args.picker,
        )
    picker_spec = spec_for(resolved_picker)
    if args.smooth_threshold is not None:
        hp["segm_first_break_smooth_threshold"] = int(args.smooth_threshold)
    hp["segm_class_count"] = picker_spec.segm_class_count or getattr(model, "segm_class_count", 1)
    hp["picker"] = resolved_picker
    hp["segm_first_break_prob_threshold"] = hp.get(
        "segm_first_break_prob_threshold",
        getattr(model, "segm_first_break_prob_threshold", 0.0),
    )
    evaluator = make_eval_evaluator(hp, resolved_picker)

    parser = train_cli.build_split_parser(
        site_names,
        prefix="valid",
        backend=args.backend,
        data_dir=args.data_dir,
        npz_root=npz_root,
        eval_ratio=eval_ratio,
        augment=False,
        use_eval_split=bool(eval_ratio),
        segm_class_count=int(hp["segm_class_count"]),
        first_break_prior=bool(getattr(model, "use_first_break_prior", False)),
    )
    collate_fn = functools.partial(
        fbp_data_module.fbp_batch_collate,
        pad_to_nearest_pow2=True,
    )
    worker_kwargs: Dict[str, Any] = {}
    if args.num_workers > 0:
        worker_kwargs["persistent_workers"] = True
        worker_kwargs["prefetch_factor"] = 2
    loader = torch.utils.data.DataLoader(
        parser,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        **worker_kwargs,
    )
    print(
        f"Picker: {resolved_picker}\n"
        f"Checkpoint: {ckpt}\n"
        f"Config: {config_path}\n"
        f"Eval sites: {site_names}  gathers={len(parser)}  batches={len(loader)}\n"
        f"Device: {device}"
    )

    traces_raw, evaluator_summary, mean_loss = run_eval(model, loader, device, evaluator)
    dt_by_origin = collect_sample_rates(parser, site_names)
    traces = annotate_trace_frame(
        traces_raw,
        origin_id_map=evaluator.origin_id_map,
        sample_rate_ms_by_origin=dt_by_origin,
    )
    metrics = headline_metrics(traces)
    metrics["loss"] = mean_loss
    metrics["picker"] = resolved_picker
    if resolved_picker == PICKER_BEFORE_AFTER:
        metrics["smooth_threshold"] = hp.get("segm_first_break_smooth_threshold")
    metrics["evaluator"] = {
        str(k): (float(v) if np.isfinite(float(v)) else None) for k, v in evaluator_summary.items()
    }
    offset_df = offset_bin_table(traces)
    gather_df = gather_summary(traces)
    worst_df, typical_df = pick_gallery_gathers(
        gather_df, n_worst=args.n_worst, n_typical=args.n_typical
    )
    high_rmse_df = (
        pick_high_rmse_gathers(gather_df, args.rmse_above)
        if args.rmse_above is not None
        else gather_df.iloc[0:0].copy()
    )

    encoder = str(hp.get("unet_encoder_type") or "model")
    if resolved_picker == PICKER_BEFORE_AFTER and "before-after" not in encoder.lower():
        encoder = f"{encoder}-before-after"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"eval_{site_label}_{str(encoder).replace('/', '-')}"
    report_root = (args.report_dir or (REPO_ROOT / "report")).resolve()
    report_dir = report_root / f"{run_name}_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = report_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    traces_path = write_trace_table(traces, report_dir / "traces")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    if not offset_df.empty:
        offset_df.to_csv(report_dir / "offset_bins.csv", index=False)
    if not gather_df.empty:
        gather_df.to_csv(report_dir / "gathers.csv", index=False)

    gather_index = build_gather_index(parser)
    worst_cards = plot_gallery(
        worst_df,
        parser=parser,
        gather_index=gather_index,
        traces=traces,
        figs_dir=figs_dir,
        prefix="worst",
        dt_by_origin=dt_by_origin,
    )
    typical_cards = plot_gallery(
        typical_df,
        parser=parser,
        gather_index=gather_index,
        traces=traces,
        figs_dir=figs_dir,
        prefix="typical",
        dt_by_origin=dt_by_origin,
    )
    high_rmse_cards: List[Dict[str, Any]] = []
    if args.rmse_above is not None:
        print(
            f"High-RMSE gallery: {len(high_rmse_df)} gather(s) with RMSE > {args.rmse_above} samples"
        )
        if len(high_rmse_df) > 200:
            logger.warning(
                "Plotting %d high-RMSE gathers; this can take a while and use a lot of disk",
                len(high_rmse_df),
            )
        high_rmse_cards = plot_gallery(
            high_rmse_df,
            parser=parser,
            gather_index=gather_index,
            traces=traces,
            figs_dir=figs_dir,
            prefix="high_rmse",
            dt_by_origin=dt_by_origin,
        )

    write_stats_html(traces, metrics, offset_df, gather_df, report_dir / "stats.html")
    write_worst_html(worst_cards, report_dir / "worst.html", title="Worst residual gathers")
    write_worst_html(typical_cards, report_dir / "typical.html", title="Typical gathers")
    if args.rmse_above is not None:
        write_worst_html(
            high_rmse_cards,
            report_dir / "high_rmse.html",
            title=f"Gathers with RMSE > {args.rmse_above} samples (reference + prediction)",
        )
    meta = {
        "run_name": run_name,
        "picker": resolved_picker,
        "checkpoint": str(ckpt),
        "model_config": str(config_path) if config_path else "",
        "encoder": encoder,
        "sites": list(site_names),
        "fold": site_label if args.fold else None,
        "backend": args.backend,
        "smooth_threshold": metrics.get("smooth_threshold"),
        "rmse_above": args.rmse_above,
    }
    write_report_md(
        report_dir / "report.md",
        meta=meta,
        metrics=metrics,
        offset_df=offset_df,
        worst_names=[c["image"] for c in worst_cards],
        typical_names=[c["image"] for c in typical_cards],
        high_rmse_names=[c["image"] for c in high_rmse_cards],
    )
    links = {
        "Markdown report": "report.md",
        "Plotly statistics": "stats.html",
        "Worst gathers": "worst.html",
        "Typical gathers": "typical.html",
        "Trace table": traces_path.name,
        "metrics.json": "metrics.json",
    }
    if args.rmse_above is not None:
        links[f"High-RMSE gathers (>{args.rmse_above})"] = "high_rmse.html"
    write_index_html(
        report_dir / "index.html",
        title=f"FBP validation — {run_name}",
        metrics=metrics,
        links=links,
    )

    print("\nHeadline:")
    for key in (
        "n_gathers",
        "n_labeled",
        "HitRate1px",
        "HitRate5px",
        "MeanAbsoluteError",
        "MeanAbsoluteErrorMs",
        "GatherCoverage",
        "loss",
    ):
        print(f"  {key:24s} {metrics.get(key)}")
    print(f"\nReport: {report_dir / 'report.md'}")
    print(f"Open:   {report_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
