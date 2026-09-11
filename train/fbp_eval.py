#!/usr/bin/env python3
"""Validate first-break picks and write a rich prediction report.

Default is an FBPUNet checkpoint. Pass ``--picker sta-lta`` to score the
Jones & van der Baan adaptive STA-LTA (no checkpoint).

    report/eval_<label>_<YYYYMMDD_HHMMSS>/
      report.md  index.html  stats.html  worst.html  typical.html
      metrics.json  traces.parquet (or traces.csv.gz)
      figs/worst_*.png  figs/typical_*.png

Example::

    python train/fbp_eval.py --ckpt-dir /home/mika/data/seismic_activity/weights/baseline/foldA \\
        --fold A --backend hdf5 --data-dir /tmp/data/
    python train/fbp_eval.py --ckpt output/train_foldA_resnet34/best-epoch=013-step=015232.ckpt \\
        --fold A --backend npz
    python train/fbp_eval.py --picker sta-lta --fold A --backend hdf5
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
    plot_gather_residual,
    write_index_html,
    write_report_md,
    write_stats_html,
    write_trace_table,
    write_worst_html,
)
from seismic_utils.hardpicks_bridge import hardpicks_available, hardpicks_item_to_shot_gather
from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat
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
        description="Validate FBPUNet or STA-LTA-OS first-break picks and write a prediction report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--picker",
        choices=("fbpunet", "sta-lta"),
        default="fbpunet",
        help="fbpunet needs --ckpt/--ckpt-dir; sta-lta is Jones & van der Baan adaptive STA-LTA.",
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
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--eval-ratio",
        type=float,
        default=None,
        help="If set with --sites, take this intra-site holdout (same splitter as training).",
    )
    p.add_argument("--n-worst", type=int, default=8, help="Worst gathers to plot.")
    p.add_argument("--n-typical", type=int, default=4, help="Typical/good gathers to plot.")
    p.add_argument("--report-dir", type=Path, default=None, help="Report root (default: <repo>/report).")
    p.add_argument("--sta-lta-th", type=float, default=1.3, help="STA-LTA-OS detection threshold Th.")
    p.add_argument(
        "--sta-lta-lw",
        type=float,
        default=0.5,
        help="STA-LTA-OS long-window length in seconds (paper 0.50 s). First-break EM uses the full trace; this still caps the short window.",
    )
    p.add_argument(
        "--sta-lta-sw",
        type=float,
        default=0.05,
        help="STA-LTA-OS short window in seconds (paper: 0.05 s at 4 kHz).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--list-folds", action="store_true")
    args = p.parse_args(argv)
    if not args.list_folds and args.picker == "fbpunet" and args.ckpt is None and args.ckpt_dir is None:
        p.error("Provide --ckpt or --ckpt-dir (or use --picker sta-lta)")
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
    model.eval()
    evaluator.reset()
    losses: List[float] = []
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="eval")):
            batch = batch_to_device(batch, device)
            _preds, loss, _metrics = model._generic_step(batch, batch_idx, evaluator)
            losses.append(float(loss.detach().cpu()))
    evaluator.finalize()
    summary = evaluator.summarize()
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return evaluator._dataframe.copy(), summary, mean_loss


def _sta_lta_options_from_args(args: argparse.Namespace):
    from seismic_utils.sta_lta import StaLtaOptions

    return StaLtaOptions(th=float(args.sta_lta_th), lw_s=float(args.sta_lta_lw), sw_s=float(args.sta_lta_sw))


def run_sta_lta_eval(parser, opts) -> tuple[pd.DataFrame, Dict[str, int], Dict[str, float]]:
    """Score STA-LTA-OS picks on every gather in *parser* (no neural net)."""
    from seismic_utils.sta_lta import pick_first_breaks

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **k: x  # noqa: E731

    origin_id_map: Dict[str, int] = {}
    dt_by_origin: Dict[str, float] = {}
    chunks: List[pd.DataFrame] = []

    for i in tqdm(range(len(parser)), desc="sta-lta"):
        item = parser[i]
        origin = str(item.get("origin") or item.get("site_name") or "unknown")
        if origin not in origin_id_map:
            origin_id_map[origin] = len(origin_id_map)
        dt_ms = sample_rate_ms_from_item(item)
        dt_by_origin.setdefault(origin, dt_ms)

        samples = np.asarray(item["samples"], dtype=np.float64)
        if samples.ndim != 2:
            raise ValueError(f"gather {i}: samples must be 2D, got {samples.shape}")
        n_tr = int(samples.shape[0])
        rec_ids = np.asarray(item["rec_ids"]).reshape(-1)
        if rec_ids.shape[0] != n_tr:
            rec_ids = np.arange(n_tr, dtype=np.int64)
        good = rec_ids != -1
        offsets_raw = item.get("offset_distances")
        if offsets_raw is not None:
            offsets = np.asarray(offsets_raw, dtype=np.float64).reshape(n_tr, -1)[:, 0]
        else:
            offsets = np.full(n_tr, np.nan, dtype=np.float64)
        target = np.asarray(item["first_break_labels"], dtype=np.float64).reshape(-1)
        if target.shape[0] != n_tr:
            target = np.resize(target, n_tr)

        pred, qual = pick_first_breaks(
            samples,
            1000.0 / max(dt_ms, 1e-12),
            opts,
            return_quality=True,
        )
        pred_int = np.zeros(n_tr, dtype=np.int64)
        finite = np.isfinite(pred) & (pred > 0)
        pred_int[finite] = np.rint(pred[finite]).astype(np.int64)
        valid_tgt = target > 0
        errors = np.where(valid_tgt, pred_int.astype(np.float64) - target, np.nan)

        idx = np.flatnonzero(good)
        if idx.size == 0:
            continue
        chunks.append(
            pd.DataFrame(
                {
                    "GatherId": np.full(idx.size, int(item["gather_id"]), dtype=np.int64),
                    "ShotId": np.full(idx.size, int(item["shot_id"]), dtype=np.int64),
                    "ReceiverId": rec_ids[idx].astype(np.int64, copy=False),
                    "OriginId": np.full(idx.size, origin_id_map[origin], dtype=np.int64),
                    "Offset": offsets[idx],
                    "Predictions": pred_int[idx],
                    "Probabilities": qual[idx],
                    "GatherCoverage": pred_int[idx] > 0,
                    "ExpectedCoverage": np.ones(idx.size, dtype=bool),
                    "Errors": errors[idx],
                }
            )
        )

    traces = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        {
            "GatherId": pd.Series(dtype="int"),
            "ShotId": pd.Series(dtype="int"),
            "ReceiverId": pd.Series(dtype="int"),
            "OriginId": pd.Series(dtype="int"),
            "Offset": pd.Series(dtype="float"),
            "Predictions": pd.Series(dtype="int"),
            "Probabilities": pd.Series(dtype="float"),
            "GatherCoverage": pd.Series(dtype="bool"),
            "ExpectedCoverage": pd.Series(dtype="bool"),
            "Errors": pd.Series(dtype="float"),
        }
    )
    return traces, origin_id_map, dt_by_origin


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
    if args.picker == "fbpunet":
        print("PL compat:", ensure_hardpicks_lightning_compat())
    if not hardpicks_available():
        raise SystemExit("hardpicks (+ torch) required — run setup_lightning.sh / install requirements")

    site_label, site_names, eval_ratio = resolve_eval_sites(args)
    npz_root = args.npz_root or (args.data_dir / "npz")
    parser = train_cli.build_split_parser(
        site_names,
        prefix="valid",
        backend=args.backend,
        data_dir=args.data_dir,
        npz_root=npz_root,
        eval_ratio=eval_ratio,
        augment=False,
        use_eval_split=bool(eval_ratio),
    )

    ckpt: Optional[Path] = None
    config_path: Optional[Path] = None
    encoder = "sta-lta" if args.picker == "sta-lta" else "model"

    if args.picker == "sta-lta":
        sta_opts = _sta_lta_options_from_args(args)
        print(
            f"Picker: STA-LTA-OS (Th={sta_opts.th}, Lw={sta_opts.lw_s}s, Sw={sta_opts.sw_s}s)\n"
            f"Eval sites: {site_names}  gathers={len(parser)}"
        )
        traces_raw, origin_id_map, dt_by_origin = run_sta_lta_eval(parser, sta_opts)
        traces = annotate_trace_frame(
            traces_raw,
            origin_id_map=origin_id_map,
            sample_rate_ms_by_origin=dt_by_origin,
        )
        metrics = headline_metrics(traces)
        metrics["loss"] = None
        metrics["picker"] = "sta-lta"
        metrics["sta_lta"] = {"th": sta_opts.th, "lw_s": sta_opts.lw_s, "sw_s": sta_opts.sw_s}
    else:
        import hardpicks.metrics.fbp.evaluator as fbp_eval
        import hardpicks.data.fbp.data_module as fbp_data_module
        import torch.utils.data

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

        model = load_fbp_model(ckpt)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.images_to_display = 0
        hp = dict(getattr(model, "hparams", {}) or {})
        if file_cfg:
            hp = {**file_cfg, **hp}
        hp["eval_metrics"] = merge_eval_metrics(hp.get("eval_metrics"))
        hp["segm_class_count"] = hp.get("segm_class_count") or getattr(model, "segm_class_count", 1)
        hp["segm_first_break_prob_threshold"] = hp.get(
            "segm_first_break_prob_threshold",
            getattr(model, "segm_first_break_prob_threshold", 0.0),
        )
        evaluator = fbp_eval.FBPEvaluator(hp)
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
        metrics["picker"] = "fbpunet"
        metrics["evaluator"] = {
            str(k): (float(v) if np.isfinite(float(v)) else None) for k, v in evaluator_summary.items()
        }
        encoder = hp.get("unet_encoder_type") or "model"
    offset_df = offset_bin_table(traces)
    gather_df = gather_summary(traces)
    worst_df, typical_df = pick_gallery_gathers(
        gather_df, n_worst=args.n_worst, n_typical=args.n_typical
    )

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

    write_stats_html(traces, metrics, offset_df, gather_df, report_dir / "stats.html")
    write_worst_html(worst_cards, report_dir / "worst.html", title="Worst residual gathers")
    write_worst_html(typical_cards, report_dir / "typical.html", title="Typical gathers")
    meta = {
        "run_name": run_name,
        "picker": args.picker,
        "checkpoint": str(ckpt) if ckpt is not None else "—",
        "model_config": str(config_path) if config_path else "",
        "encoder": encoder,
        "sites": list(site_names),
        "fold": site_label if args.fold else None,
        "backend": args.backend,
    }
    write_report_md(
        report_dir / "report.md",
        meta=meta,
        metrics=metrics,
        offset_df=offset_df,
        worst_names=[c["image"] for c in worst_cards],
        typical_names=[c["image"] for c in typical_cards],
    )
    write_index_html(
        report_dir / "index.html",
        title=f"FBP validation — {run_name}",
        metrics=metrics,
        links={
            "Markdown report": "report.md",
            "Plotly statistics": "stats.html",
            "Worst gathers": "worst.html",
            "Typical gathers": "typical.html",
            "Trace table": traces_path.name,
            "metrics.json": "metrics.json",
        },
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
