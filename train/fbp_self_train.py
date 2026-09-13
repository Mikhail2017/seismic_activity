#!/usr/bin/env python3
"""Site-specific Meneses self-training (1% labels + optional iterative expansion).

Does not use ``--fold`` or ``before_after``. Inner Lightning fits are fresh
5-epoch (or 25-epoch static) trainers. See TRAINING.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent
for path in (str(REPO_ROOT), str(TRAIN_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import fbp_eval as eval_cli
import fbp_train as train_cli
from seismic_utils.dataset import DEFAULT_DATA_DIR
from seismic_utils.fbp_eval_report import (
    MIN_GALLERY_LABELED,
    annotate_trace_frame,
    gather_summary,
    headline_metrics,
    offset_bin_table,
    pick_gallery_gathers,
    write_index_html,
    write_report_md,
    write_stats_html,
    write_trace_table,
    write_worst_html,
)
from seismic_utils.hardpicks_bridge import hardpicks_available
from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat
from seismic_utils.minimal_preprocess import (
    LABEL_WINDOW_MS,
    MinimalAnnotationDataset,
    minimal_batch_collate,
)
from seismic_utils.minimal_split import (
    collect_valid_keys,
    gather_key,
    indices_for_keys,
    key_to_dict,
    load_split,
    make_minimal_split,
    split_keys,
    write_split,
)
from seismic_utils.pickers import (
    PICKER_BEFORE_AFTER,
    attach_unpicked_evaluators,
    decode_argmax_fb_unpicked,
    parse_model_suffixes,
)
from seismic_utils.predict import resolve_checkpoint
from seismic_utils.pseudo_label_qc import draw_without_replacement, qc_gather_picks
from seismic_utils.training_state import load_checkpoint, reserve_run_directory, run_token
from seismic_utils.validation import IndexedValidationDataset

logger = logging.getLogger("fbp_self_train")

ABLATIONS = {
    "control": {"window": False, "weight": [1.0, 1.0], "iterative": False},
    "windowed": {"window": True, "weight": [1.0, 1.0], "iterative": False},
    "weighted": {"window": False, "weight": [1.0, 100.0], "iterative": False},
    "combined": {"window": True, "weight": [1.0, 100.0], "iterative": False},
    "iterative": {"window": True, "weight": [1.0, 100.0], "iterative": True},
}

DEFAULT_CONFIG = REPO_ROOT / "configs" / "minimal_annotations.yaml"
RESET_AFTER = {5, 10}
N_ITERS = 15
N_DRAW = 200
INNER_EPOCHS_ITER = 5
INNER_EPOCHS_STATIC = 25
HEADLINE_KEYS = (
    "n_gathers",
    "n_labeled",
    "HitRate1px",
    "HitRate5px",
    "MeanAbsoluteError",
    "MeanAbsoluteErrorMs",
    "GatherCoverage",
    "Coverage",
    "W_pred_0",
    "W_pred_2",
    "W_pred_5",
    "W_pred_10",
    "W_total_0",
    "W_total_2",
    "W_total_5",
    "W_total_10",
    "MAE",
    "loss",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Meneses-style 1% labelled / self-training U-Net (one site).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--sites", required=False, default=None, help="Single site name.")
    p.add_argument("--ablation", choices=tuple(ABLATIONS), default="combined")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backend", choices=("npz", "hdf5"), default=None)
    p.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    p.add_argument("--npz-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--split-json", type=Path, default=None, help="Reuse a saved split.")
    p.add_argument("--n-labeled", type=int, default=None, help="Override paper labelled count.")
    p.add_argument("--n-draw", type=int, default=N_DRAW)
    p.add_argument("--n-iters", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="Inner-fit epochs (override).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--devices", default="auto")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--precision", default=None)
    p.add_argument("--smoke", action="store_true", help="Tiny Halfmile loop for laptops.")
    p.add_argument("--n-worst", type=int, default=8, help="Worst 99%% gathers to plot.")
    p.add_argument("--n-typical", type=int, default=4, help="Typical 99%% gathers to plot.")
    p.add_argument(
        "--min-labeled",
        type=int,
        default=MIN_GALLERY_LABELED,
        help="Minimum labeled traces for a gather to appear in worst/typical galleries.",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Eval report directory (default: <output-dir>/report).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _load_recipe(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"recipe must be a mapping: {path}")
    return data


def _refuse_incompatible(recipe: Mapping[str, Any], args: argparse.Namespace) -> None:
    model = str(recipe.get("model") or "meneses")
    _, is_ba, is_horizon = parse_model_suffixes(model)
    picker = recipe.get("picker")
    if is_ba or str(picker or "") == PICKER_BEFORE_AFTER:
        raise SystemExit("minimal-annotation recipe cannot use before_after")
    if is_horizon:
        raise SystemExit("minimal-annotation recipe cannot use -horizon")
    geo = recipe.get("geonorm")
    if geo not in (None, False, 0, "0", "false", "False", "none", "null", ""):
        raise SystemExit("minimal-annotation recipe cannot enable GeoNorm")
    if getattr(args, "fold", None):
        raise SystemExit("--fold is incompatible with fbp_self_train.py")


def _draw_rng(site: str, seed: int, iteration: int) -> np.random.Generator:
    material = f"{site}|{int(seed)}|{int(iteration)}".encode()
    digest = hashlib.sha256(material).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _one_site(args: argparse.Namespace, recipe: Mapping[str, Any]) -> str:
    raw = args.sites or recipe.get("sites") or "Halfmile"
    if isinstance(raw, (list, tuple)):
        names = [str(s).strip() for s in raw if str(s).strip()]
    else:
        names = [s.strip() for s in str(raw).split(",") if s.strip()]
    if len(names) != 1:
        raise SystemExit("self-training is site-specific; pass exactly one --sites name")
    return names[0]


def _apply_ablation(model_config: dict[str, Any], ablation: str, window_ms: float) -> dict[str, Any]:
    spec = ABLATIONS[ablation]
    cfg = dict(model_config)
    loss_params = dict(cfg.get("loss_params") or {})
    loss_params["weight"] = list(spec["weight"])
    cfg["loss_params"] = loss_params
    mini = dict(cfg.get("minimal_annotations") or {})
    mini.update(
        {
            "preprocess": "trace_zscore_int16_pad1",
            "pad_multiple": 16,
            "pad_value": 1,
            "label_window_ms": float(window_ms),
            "fb_weight": float(spec["weight"][1]),
            "ablation": ablation,
        }
    )
    cfg["minimal_annotations"] = mini
    cfg["use_dist_offsets"] = False
    cfg["picker"] = "fbpunet"
    cfg["segm_class_count"] = 1
    return cfg


def _make_loader(dataset, batch_size: int, num_workers: int, *, shuffle: bool, pin_memory: bool):
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=minimal_batch_collate,
        pin_memory=pin_memory,
        **kwargs,
    )


def _inner_fit(
    *,
    model_config: dict[str, Any],
    train_ds,
    valid_ds,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    num_workers: int,
    init_ckpt: Optional[Path],
    devices,
    accelerator: str,
    precision,
    seed: int,
) -> Optional[Path]:
    import models.fbp.unet as fbp_unet

    pl.seed_everything(seed, workers=True)
    reserve_run_directory(output_dir)
    cfg = dict(model_config)
    cfg["max_epochs"] = int(epochs)
    (output_dir / "model_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    use_gpu = torch.cuda.is_available() and str(accelerator).lower() != "cpu"
    train_loader = _make_loader(train_ds, batch_size, num_workers, shuffle=True, pin_memory=use_gpu)
    valid_loader = _make_loader(
        IndexedValidationDataset(valid_ds),
        batch_size,
        num_workers,
        shuffle=False,
        pin_memory=use_gpu,
    )
    model = fbp_unet.FBPUNet(cfg)
    if init_ckpt is not None:
        blob = load_checkpoint(resolve_checkpoint(init_ckpt))
        model.load_state_dict(blob["state_dict"], strict=True)
    attach_unpicked_evaluators(model, cfg)

    tbx = pl.loggers.TensorBoardLogger(save_dir=str(output_dir / "tensorboard"), name="default", version=0, default_hp_metric=False)
    csv_logger = pl.loggers.CSVLogger(save_dir=str(output_dir / "csv_logs"), name="metrics", version=0)
    ckpt_cb = train_cli.make_model_checkpoint(output_dir, save_top_k=1)
    trainer = train_cli.make_trainer(
        tbx_logger=tbx,
        csv_logger=csv_logger,
        callbacks=[ckpt_cb],
        max_epochs=int(epochs),
        log_every_n_steps=10,
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        precision=precision,
        find_unused_parameters=False,
    )
    trainer.fit(model, train_loader, valid_loader)
    best = Path(ckpt_cb.best_model_path).resolve() if ckpt_cb.best_model_path else None
    return best if best and best.is_file() else None


def _infer_qc(model, dataset, device: torch.device) -> tuple[dict, list[dict[str, Any]]]:
    accepted: dict = {}
    stats: list[dict[str, Any]] = []
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            batch = minimal_batch_collate([item])
            tensors = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            logits = model(model._prepare_input_features(tensors))
            picks, _ = decode_argmax_fb_unpicked(logits)
            n = int(item.get("trace_count") or item["samples"].shape[0])
            pred = picks[0, :n].detach().cpu().numpy().astype(np.float64)
            pred[pred <= 0] = np.nan
            offs = item.get("offset_distances")
            off = np.asarray(offs, dtype=np.float64).reshape(n, -1)[:n, 0] if offs is not None else np.full(n, np.nan)
            qc = qc_gather_picks(off, pred)
            key = gather_key(item)
            rec = {"key": list(key), "admit": qc["admit"], "survive_frac": qc["survive_frac"]}
            stats.append(rec)
            if qc["admit"]:
                accepted[key] = qc["picks"][:n]
    return accepted, stats


def _eval_oracle(model, dataset, device: torch.device, dt_by_origin: dict[str, float]):
    from seismic_utils.pickers import make_eval_evaluator

    hp = dict(getattr(model, "hparams", {}) or {})
    evaluator = make_eval_evaluator(hp, "fbpunet")
    loader = _make_loader(dataset, batch_size=1, num_workers=0, shuffle=False, pin_memory=False)
    traces_raw, summary, mean_loss = eval_cli.run_eval(model, loader, device, evaluator)
    traces = annotate_trace_frame(
        traces_raw,
        origin_id_map=evaluator.origin_id_map,
        sample_rate_ms_by_origin=dt_by_origin,
    )
    metrics = headline_metrics(traces)
    metrics["loss"] = mean_loss
    metrics["picker"] = "fbpunet"
    metrics["evaluator"] = {
        str(k): (float(v) if np.isfinite(float(v)) else None) for k, v in summary.items()
    }
    return traces, metrics


def _write_99pct_report(
    *,
    traces,
    metrics: dict[str, Any],
    parser,
    site: str,
    ablation: str,
    seed: int,
    ckpt: Path,
    report_dir: Path,
    n_worst: int,
    n_typical: int,
    min_labeled: int,
    dt_by_origin: dict[str, float],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = report_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    offset_df = offset_bin_table(traces)
    gather_df = gather_summary(traces)
    n_with_mae = int(gather_df["MAE"].notna().sum()) if "MAE" in gather_df.columns else 0
    worst_df, typical_df = pick_gallery_gathers(
        gather_df,
        n_worst=n_worst,
        n_typical=n_typical,
        min_labeled=min_labeled,
    )
    if "n_labeled" in gather_df.columns and min_labeled > 0:
        n_eligible = int(
            ((gather_df["MAE"].notna()) & (gather_df["n_labeled"] >= min_labeled)).sum()
        )
        dropped = n_with_mae - n_eligible
        if dropped:
            logger.info(
                "Gallery: dropped %d gather(s) with n_labeled < %d",
                dropped,
                min_labeled,
            )

    traces_path = write_trace_table(traces, report_dir / "traces")
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    if not offset_df.empty:
        offset_df.to_csv(report_dir / "offset_bins.csv", index=False)
    if not gather_df.empty:
        gather_df.to_csv(report_dir / "gathers.csv", index=False)

    gather_index = eval_cli.build_gather_index(parser)
    worst_cards = eval_cli.plot_gallery(
        worst_df,
        parser=parser,
        gather_index=gather_index,
        traces=traces,
        figs_dir=figs_dir,
        prefix="worst",
        dt_by_origin=dt_by_origin,
        limit=n_worst,
    )
    typical_cards = eval_cli.plot_gallery(
        typical_df,
        parser=parser,
        gather_index=gather_index,
        traces=traces,
        figs_dir=figs_dir,
        prefix="typical",
        dt_by_origin=dt_by_origin,
        limit=n_typical,
    )

    write_stats_html(traces, metrics, offset_df, gather_df, report_dir / "stats.html")
    write_worst_html(worst_cards, report_dir / "worst.html", title="Worst residual gathers (99% pool)")
    write_worst_html(typical_cards, report_dir / "typical.html", title="Typical gathers (99% pool)")
    run_name = f"self_train_{site.lower()}_{ablation}_seed{seed}_99pct"
    cfg_path = ckpt.parent / "model_config.yaml"
    encoder = "meneses"
    if cfg_path.is_file():
        loaded = yaml.safe_load(cfg_path.read_text()) or {}
        if isinstance(loaded, dict):
            encoder = str(loaded.get("unet_encoder_type") or loaded.get("model") or encoder)
    write_report_md(
        report_dir / "report.md",
        meta={
            "run_name": run_name,
            "picker": "fbpunet",
            "checkpoint": str(ckpt),
            "model_config": str(cfg_path) if cfg_path.is_file() else "",
            "encoder": encoder,
            "sites": [site],
            "fold": "99pct-manual",
            "backend": "minimal_annotations",
            "smooth_threshold": None,
            "rmse_above": None,
            "lateral_clean": None,
        },
        metrics=metrics,
        offset_df=offset_df,
        worst_names=[c["image"] for c in worst_cards],
        typical_names=[c["image"] for c in typical_cards],
        high_rmse_names=(),
    )
    write_index_html(
        report_dir / "index.html",
        title=f"FBP self-train 99% — {run_name}",
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
    return report_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    recipe = _load_recipe(args.config)
    _refuse_incompatible(recipe, args)
    if args.smoke:
        args.sites = args.sites or "Halfmile"
        args.n_labeled = args.n_labeled or 8
        args.n_draw = min(args.n_draw or 8, 8)
        args.n_iters = args.n_iters or 2
        args.epochs = args.epochs or 1
        args.ablation = "iterative" if args.ablation == "combined" else args.ablation
        recipe.setdefault("batch_size", 2)
        recipe.setdefault("num_workers", 0)

    site = _one_site(args, recipe)
    seed = int(args.seed if args.seed is not None else recipe.get("seed", 0))
    ablation = args.ablation
    spec = ABLATIONS[ablation]
    window_ms = LABEL_WINDOW_MS if spec["window"] else 0.0
    iterative = bool(spec["iterative"])
    n_iters = int(args.n_iters if args.n_iters is not None else (N_ITERS if iterative else 1))
    inner_epochs = int(
        args.epochs
        if args.epochs is not None
        else (INNER_EPOCHS_ITER if iterative else INNER_EPOCHS_STATIC)
    )
    backend = str(args.backend or recipe.get("backend") or "hdf5")
    batch_size = int(args.batch_size or recipe.get("batch_size") or 8)
    num_workers = int(args.num_workers if args.num_workers is not None else recipe.get("num_workers") or 0)
    precision = args.precision or recipe.get("precision") or "32"

    print("PL compat:", ensure_hardpicks_lightning_compat())
    if not hardpicks_available():
        raise SystemExit("hardpicks (+ torch) required")

    model_config, model_label = train_cli.build_model_config(
        model=str(recipe.get("model") or "meneses"),
        max_epochs=inner_epochs,
        recipe=recipe,
        picker="fbpunet",
    )
    model_config = _apply_ablation(model_config, ablation, window_ms)

    npz_root = args.npz_root or (args.data_dir / "npz")
    parser = train_cli.build_split_parser(
        [site],
        prefix="train",
        backend=backend,
        data_dir=args.data_dir,
        npz_root=npz_root,
        eval_ratio=None,
        augment=False,
        use_eval_split=False,
        segm_class_count=1,
        extra_site_params={"normalize_samples": False, "segm_first_break_buffer": 0},
    )
    if args.split_json:
        split = load_split(args.split_json)
    else:
        valid_keys = collect_valid_keys(parser)
        split = make_minimal_split(valid_keys, site=site, seed=seed, n_labeled=args.n_labeled)

    labeled_train = split_keys(split, "labeled_train")
    labeled_val = split_keys(split, "labeled_val")
    unlabeled_pool = split_keys(split, "unlabeled_pool")
    train_idx = indices_for_keys(parser, labeled_train)
    val_idx = indices_for_keys(parser, labeled_val)
    pool_idx = indices_for_keys(parser, unlabeled_pool)

    stamp = run_token(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out_root = (args.output_dir or (REPO_ROOT / "output" / f"self_train_{site.lower()}_{ablation}_{stamp}")).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    write_split(split, out_root / "split.json")

    labeled_ds = MinimalAnnotationDataset(parser, train_idx, mode="labeled", window_ms=window_ms)
    valid_ds = MinimalAnnotationDataset(parser, val_idx, mode="oracle", window_ms=window_ms)
    test_ds = MinimalAnnotationDataset(parser, pool_idx, mode="oracle", window_ms=window_ms)

    devices = train_cli.parse_devices(args.devices)
    remaining = list(unlabeled_pool)
    accepted_pseudo: dict = {}
    last_ckpt: Optional[Path] = None
    history: List[dict[str, Any]] = []

    print(
        f"Site {site} ablation={ablation} seed={seed} labelled={len(labeled_train)}/{len(labeled_val)} "
        f"pool={len(unlabeled_pool)} iters={n_iters} epochs/iter={inner_epochs}"
    )

    for it in range(1, n_iters + 1):
        reset = last_ckpt is None and it > 1
        init = last_ckpt
        iter_dir = out_root / f"iter_{it:02d}"
        pseudo_keys = list(accepted_pseudo.keys())
        parts = [labeled_ds]
        if pseudo_keys:
            pidx = indices_for_keys(parser, pseudo_keys)
            parts.append(
                MinimalAnnotationDataset(
                    parser, pidx, mode="pseudo", window_ms=window_ms, pseudo_picks=accepted_pseudo
                )
            )
        if len(parts) == 1:
            train_ds = parts[0]
        else:
            from hardpicks.data.fbp.gather_wrappers import ShotLineGatherConcatDataset

            train_ds = ShotLineGatherConcatDataset(parts)

        print(f"\n=== iteration {it}/{n_iters} train_gathers={len(train_ds)} reset={reset} ===")
        best = _inner_fit(
            model_config=model_config,
            train_ds=train_ds,
            valid_ds=valid_ds,
            output_dir=iter_dir,
            epochs=inner_epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            init_ckpt=init,
            devices=devices,
            accelerator=args.accelerator,
            precision=precision,
            seed=seed + it,
        )
        last_ckpt = best or last_ckpt
        rec = {"iteration": it, "n_train": len(train_ds), "ckpt": str(best) if best else None}

        if iterative and it < n_iters and remaining:
            if last_ckpt is None:
                raise SystemExit(f"iteration {it} produced no checkpoint; cannot expand labels")
            n_draw = min(int(args.n_draw), len(remaining))
            rng = _draw_rng(site, seed, it)
            drawn, remaining = draw_without_replacement(remaining, n_draw, rng)
            draw_idx = indices_for_keys(parser, drawn)
            infer_ds = MinimalAnnotationDataset(parser, draw_idx, mode="unlabeled", window_ms=window_ms)
            import models.fbp.unet as fbp_unet

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = fbp_unet.FBPUNet.load_from_checkpoint(str(last_ckpt), map_location="cpu")
            attach_unpicked_evaluators(model, dict(model.hparams))
            new_acc, qc_stats = _infer_qc(model, infer_ds, device)
            accepted_pseudo.update(new_acc)
            rec.update(
                {
                    "n_drawn": len(drawn),
                    "n_admitted": len(new_acc),
                    "n_pseudo_total": len(accepted_pseudo),
                    "drawn": [key_to_dict(k) for k in drawn],
                    "remaining": [key_to_dict(k) for k in remaining],
                    "qc": qc_stats,
                }
            )
            if it in RESET_AFTER:
                last_ckpt = None
                rec["weight_reset"] = True
                print(f"Weight reset after iteration {it}")
        iter_name = f"iter_{it:02d}.json"
        payload = json.dumps(rec, indent=2, default=str) + "\n"
        (iter_dir / "iter.json").write_text(payload)
        (out_root / iter_name).write_text(payload)
        history.append({k: v for k, v in rec.items() if k not in {"qc", "drawn", "remaining"}})

    metrics = {}
    report_dir = None
    if last_ckpt is not None:
        import models.fbp.unet as fbp_unet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = fbp_unet.FBPUNet.load_from_checkpoint(str(last_ckpt), map_location="cpu")
        attach_unpicked_evaluators(model, dict(model.hparams))
        dt_by_origin = eval_cli.collect_sample_rates(parser, [site])
        print("\nEvaluating 99% pool against manual picks…")
        traces, metrics = _eval_oracle(model.to(device), test_ds, device, dt_by_origin)
        print("Headline:")
        for key in HEADLINE_KEYS:
            print(f"  {key:24s} {metrics.get(key)}")
        report_dir = (args.report_dir or (out_root / "report")).resolve()
        _write_99pct_report(
            traces=traces,
            metrics=metrics,
            parser=parser,
            site=site,
            ablation=ablation,
            seed=seed,
            ckpt=last_ckpt,
            report_dir=report_dir,
            n_worst=int(args.n_worst),
            n_typical=int(args.n_typical),
            min_labeled=int(args.min_labeled),
            dt_by_origin=dt_by_origin,
        )
        print(f"\nReport: {report_dir / 'report.md'}")
        print(f"Open:   {report_dir / 'index.html'}")

    state = {
        "site": site,
        "ablation": ablation,
        "seed": seed,
        "history": history,
        "n_pseudo": len(accepted_pseudo),
        "best_ckpt": str(last_ckpt) if last_ckpt else None,
        "metrics_99pct": metrics,
        "report_dir": str(report_dir) if report_dir else None,
    }
    (out_root / "self_train_state.json").write_text(json.dumps(state, indent=2, default=str) + "\n")
    print("Wrote", out_root / "self_train_state.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
