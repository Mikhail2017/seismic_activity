#!/usr/bin/env python3
"""Train local FBPUNet (cloned from hardpicks) on NPZ or HDF5 gathers.

Reports to TensorBoard + CSV, prints step/epoch loss and validation metrics, and
saves the best checkpoint on ``valid/HitRate1px``.

The trainer class lives in ``models.fbp.unet.FBPUNet``. Different “models” are
encoder/architecture presets (ResNet18, EfficientNet-B0, …) or a YAML/JSON
hyperparameter file. Optional SMP backbone pretraining via ``--encoder-weights``.

Example::

    conda activate seismic_activity
    python examples/local/fbp_train.py --sites Brunswick --model resnet18 --epochs 5
    python examples/local/fbp_train.py --sites Brunswick --model efficientnet-b0 \\
        --encoder-weights imagenet
    python examples/local/fbp_train.py --sites Brunswick --model-config my_model.yaml

    tensorboard --logdir output/train_brunswick_resnet18/tensorboard
"""

from __future__ import annotations

import argparse
import copy
import functools
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seismic_utils.dataset import DEFAULT_DATA_DIR
from seismic_utils.hardpicks_bridge import hardpicks_available, resolve_hardpicks_site_info
from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat
from seismic_utils.npz_parser import create_npz_parser

logger = logging.getLogger("fbp_train")

MONITOR_METRIC = "valid/HitRate1px"
SEGMENTATION_CLASS_COUNT = 1

# Named architecture presets (all use local models.fbp.unet.FBPUNet). Decoder / LR
# follow hardpicks fold configs where available. Keys are CLI --model names.
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "unet_encoder_type": "resnet18",
        "encoder_block_count": 5,
        "mid_block_channels": 0,
        "decoder_block_channels": "[256, 128, 64, 32, 16]",
        "lr": 0.002136,
    },
    "resnet34": {
        "unet_encoder_type": "resnet34",
        "encoder_block_count": 5,
        "mid_block_channels": 0,
        "decoder_block_channels": "[256, 128, 64, 32, 16]",
        "lr": 0.002136,
    },
    "resnet50": {
        "unet_encoder_type": "resnet50",
        "encoder_block_count": 5,
        "mid_block_channels": 0,
        "decoder_block_channels": "[512, 256, 128, 64, 32]",
        "lr": 0.0015,
    },
    "efficientnet-b0": {
        "unet_encoder_type": "timm-efficientnet-b0",
        "encoder_block_count": 5,
        "mid_block_channels": 0,
        "decoder_block_channels": "[512, 256, 128, 64, 32]",
        "lr": 0.003417,
    },
    "efficientnet-b4": {
        "unet_encoder_type": "timm-efficientnet-b4",
        "encoder_block_count": 5,
        "mid_block_channels": 0,
        "decoder_block_channels": "[512, 256, 128, 64, 32]",
        "lr": 0.001053,
    },
    "vanilla": {
        "unet_encoder_type": "vanilla",
        "encoder_block_count": 5,
        "mid_block_channels": 256,
        "decoder_block_channels": "[256, 128, 64, 32, 16]",
        "lr": 0.001,
    },
}

# Allow hardpicks / SMP encoder spellings to resolve to the same preset.
MODEL_ALIASES: Dict[str, str] = {
    "timm-efficientnet-b0": "efficientnet-b0",
    "timm-efficientnet-b4": "efficientnet-b4",
    "fbpunet": "resnet18",
    "fbp-unet": "resnet18",
}

TRAIN_AUGMENTATIONS = [
    {
        "type": "crop",
        "params": {
            "low_sample_count": 512,
            "high_sample_count": 1024,
            "max_crop_fraction": 0.333,
        },
    },
    {"type": "flip"},
]

COMMON_SITE_PARAMS = {
    "normalize_samples": True,
    "segm_first_break_buffer": 0,
}

# Keys that belong to train-script CLI / presets but are not FBPUNet hyperparams.
_MODEL_CONFIG_META_KEYS = frozenset({"lr", "model_name", "preset"})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    preset_names = ", ".join(sorted(MODEL_PRESETS))
    p = argparse.ArgumentParser(
        description="Train FBPUNet first-break picker (NPZ or HDF5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sites",
        default="Brunswick,Halfmile",
        help="Comma-separated site names (Brunswick | Halfmile | Lalor | Sudbury).",
    )
    p.add_argument(
        "--backend",
        choices=("npz", "hdf5"),
        default="npz",
        help="Data backend: NPZ (fast) or live HDF5.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help="Root directory for seismic HDF5 assets.",
    )
    p.add_argument(
        "--npz-root",
        type=Path,
        default=None,
        help="NPZ root (default: <data-dir>/npz).",
    )
    p.add_argument("--epochs", type=int, default=5, help="Max training epochs.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--eval-ratio", type=float, default=0.15, help="Per-site validation fraction.")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Experiment directory (default: output/train_<sites>_<model>).",
    )
    p.add_argument(
        "--model",
        default="resnet18",
        help=(
            "Architecture preset name, or any SMP encoder id "
            f"(presets: {preset_names})."
        ),
    )
    p.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="YAML/JSON file of FBPUNet hyperparams (merged over the preset).",
    )
    p.add_argument(
        "--encoder-weights",
        default=None,
        help=(
            "SMP backbone pretrained weights (e.g. imagenet, ssl, swsl). "
            "Default: train encoder from scratch. Ignored for vanilla encoder."
        ),
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model presets and exit.",
    )
    p.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Lightning / TensorBoard step logging interval.",
    )
    p.add_argument(
        "--print-every-n-steps",
        type=int,
        default=50,
        help="Print train loss to stdout every N steps (0 disables).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: preset-specific, else 0.002136).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-final-validate",
        action="store_true",
        help="Skip re-validation with the best checkpoint after fit.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def resolve_model_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if key in MODEL_PRESETS:
        return key
    # Accept SMP / hardpicks encoder ids as custom model labels.
    return name.strip()


def _load_config_file(path: Path) -> Dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model config not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        # Try YAML first, then JSON.
        try:
            data = yaml.safe_load(text)
        except Exception:
            data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"model config must be a mapping, got {type(data).__name__}: {path}")
    # Allow wrapping under a top-level "model:" key (common in larger configs).
    if "unet_encoder_type" not in data and "encoder_type" not in data and isinstance(data.get("model"), dict):
        data = data["model"]
    return data


def _decoder_for_encoder(encoder_type: str) -> Dict[str, Any]:
    """Best-effort decoder defaults when --model is a raw SMP encoder name."""
    enc = encoder_type.lower()
    if enc.startswith("resnet") and not enc.startswith("resnet5") and "101" not in enc and "152" not in enc:
        return {
            "encoder_block_count": 5,
            "mid_block_channels": 0,
            "decoder_block_channels": "[256, 128, 64, 32, 16]",
            "lr": 0.002136,
        }
    if "efficientnet" in enc or enc.startswith("resnet"):
        return {
            "encoder_block_count": 5,
            "mid_block_channels": 0,
            "decoder_block_channels": "[512, 256, 128, 64, 32]",
            "lr": 0.0015,
        }
    if enc == "vanilla":
        return {
            "encoder_block_count": 5,
            "mid_block_channels": 256,
            "decoder_block_channels": "[256, 128, 64, 32, 16]",
            "lr": 0.001,
        }
    return {
        "encoder_block_count": 5,
        "mid_block_channels": 0,
        "decoder_block_channels": "[512, 256, 128, 64, 32]",
        "lr": 0.0015,
    }


def build_model_config(
    *,
    model: str,
    max_epochs: int,
    lr: Optional[float] = None,
    model_config_path: Optional[Path] = None,
    encoder_weights: Optional[str] = None,
) -> tuple[Dict[str, Any], str]:
    """Build FBPUNet hyperparams from a preset and/or YAML/JSON override.

    Returns ``(hyper_params, model_label)`` where ``model_label`` is used in
    output paths and logs.
    """
    model_label = resolve_model_name(model)
    if model_label in MODEL_PRESETS:
        arch = copy.deepcopy(MODEL_PRESETS[model_label])
    else:
        # Treat as a raw encoder id (must be known to SMP at construct time).
        arch = {"unet_encoder_type": model_label, **_decoder_for_encoder(model_label)}
        model_label = model_label.replace("/", "-")

    file_overrides: Dict[str, Any] = {}
    if model_config_path is not None:
        file_overrides = _load_config_file(model_config_path)
        # If the file defines the encoder, prefer that for the label when no preset matched.
        enc = file_overrides.get("unet_encoder_type") or file_overrides.get("encoder_type")
        if enc and model.strip().lower() in {"", "custom"}:
            model_label = str(enc)

    preset_lr = float(arch.pop("lr", 0.002136))
    file_lr = file_overrides.pop("lr", None)
    if isinstance(file_lr, dict):
        file_lr = None  # ignore accidental nested structures
    # optimizer_params.lr in the file still wins later via deep merge below.

    base = {
        "model_type": "FBPUNet",
        "unet_decoder_type": "vanilla",
        "decoder_attention_type": None,
        "segm_class_count": SEGMENTATION_CLASS_COUNT,
        "use_dist_offsets": True,
        "use_first_break_prior": False,
        "coordconv": False,
        "encoder_weights": None,
        "optimizer_type": "Adam",
        "optimizer_params": {"lr": preset_lr, "weight_decay": 1e-6},
        "scheduler_type": "StepLR",
        "scheduler_params": {"step_size": 10, "gamma": 0.1},
        "update_scheduler_at_epochs": True,
        "loss_type": "crossentropy",
        "loss_params": {},
        "use_full_metrics_during_training": False,
        "eval_type": "FBPEvaluator",
        "segm_first_break_prob_threshold": 0.0,
        "eval_metrics": [
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 1}},
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 3}},
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 5}},
            {"metric_type": "MeanBiasError"},
            {"metric_type": "MeanAbsoluteError"},
        ],
        "gathers_to_display": 0,
        "use_checkpointing": False,
        "max_epochs": max_epochs,
    }
    base.update(arch)

    # Merge file overrides (shallow + nested optimizer_params / scheduler_params).
    for key, value in file_overrides.items():
        if key in _MODEL_CONFIG_META_KEYS:
            continue
        if key in {"optimizer_params", "scheduler_params", "loss_params"} and isinstance(value, dict):
            merged = dict(base.get(key) or {})
            merged.update(value)
            base[key] = merged
        else:
            base[key] = value

    # CLI --lr always wins when provided.
    if lr is not None:
        opt = dict(base.get("optimizer_params") or {})
        opt["lr"] = float(lr)
        base["optimizer_params"] = opt
    elif file_lr is not None:
        opt = dict(base.get("optimizer_params") or {})
        opt["lr"] = float(file_lr)
        base["optimizer_params"] = opt

    # CLI --encoder-weights wins over preset/file when provided.
    if encoder_weights is not None:
        ew = encoder_weights.strip()
        base["encoder_weights"] = None if ew.lower() in {"", "none", "null"} else ew

    base["max_epochs"] = max_epochs
    base["model_type"] = base.get("model_type") or "FBPUNet"
    if base["model_type"] != "FBPUNet":
        raise ValueError(
            f"Only model_type=FBPUNet is supported "
            f"(got {base['model_type']!r}). Use --model / unet_encoder_type for architectures."
        )
    return base, model_label


def list_models() -> None:
    print("Available --model presets (all train local models.fbp.unet.FBPUNet):\n")
    for name, cfg in sorted(MODEL_PRESETS.items()):
        print(
            f"  {name:18s}  encoder={cfg['unet_encoder_type']}"
            f"  decoder={cfg['decoder_block_channels']}  lr={cfg['lr']}"
        )
    print("\nAliases:", ", ".join(f"{k}->{v}" for k, v in sorted(MODEL_ALIASES.items())))
    print(
        "\nAny segmentation_models_pytorch encoder name is also accepted "
        "(decoder channels are inferred)."
    )
    print("Pretrained backbones: --encoder-weights imagenet  (or ssl/swsl/… per encoder)")
    print("Override anything with --model-config path/to.yaml")


def _gather_key(meta: dict) -> str:
    """Site-qualified gather identity (gather_id alone collides across sites)."""
    return (
        f"{meta['origin']}_g{int(meta['gather_id'])}"
        f"_s{int(meta['shot_id'])}_r{int(meta['rec_line_id'])}"
    )


def _concat_or_single(parts: list):
    from hardpicks.data.fbp.gather_wrappers import ShotLineGatherConcatDataset

    assert parts, "expected at least one site parser"
    if len(parts) == 1:
        return parts[0]
    return ShotLineGatherConcatDataset(parts)


def build_parsers(
    site_names: Sequence[str],
    backend: str,
    data_dir: Path,
    npz_root: Path,
    eval_ratio: float,
):
    import hardpicks
    import hardpicks.data.fbp.data_module as fbp_data_module

    train_parts: list = []
    valid_parts: list = []
    backend = backend.strip().lower()

    if backend == "npz":
        for site_name in site_names:
            logger.info("NPZ parser: %s", npz_root / site_name)
            train_parts.append(
                create_npz_parser(
                    site_name,
                    npz_root=npz_root,
                    prefix="train",
                    site_params={
                        **COMMON_SITE_PARAMS,
                        "augmentations": TRAIN_AUGMENTATIONS,
                        "subset": {"eval_ratio": eval_ratio, "use_eval_split": False},
                    },
                    segm_class_count=SEGMENTATION_CLASS_COUNT,
                )
            )
            valid_parts.append(
                create_npz_parser(
                    site_name,
                    npz_root=npz_root,
                    prefix="valid",
                    site_params={
                        **COMMON_SITE_PARAMS,
                        "subset": {"eval_ratio": eval_ratio, "use_eval_split": True},
                    },
                    segm_class_count=SEGMENTATION_CLASS_COUNT,
                )
            )
    elif backend == "hdf5":
        rejected = Path(hardpicks.FBP_BAD_GATHERS_DIR) / "bad-gather-ids_combined.yaml"
        if not rejected.is_file():
            rejected = None
            logger.warning("bad-gather YAML not found; continuing without reject list")

        hdf5_site_params = {
            **COMMON_SITE_PARAMS,
            "rejected_gather_yaml_path": str(rejected) if rejected else None,
            "use_cache": False,
        }
        generic_site_params = dict(
            convert_to_fp16=True,
            convert_to_int16=True,
            preload_trace_data=False,
            cache_trace_metadata=True,
            provide_offset_dists=True,
        )
        for site_name in site_names:
            site_info = resolve_hardpicks_site_info(site_name, data_dir=data_dir)
            logger.info("HDF5 parser: %s", site_name)
            for k, v in site_info.items():
                logger.info("  %s: %s", k, v)
            train_parts.append(
                fbp_data_module.FBPDataModule.create_parser(
                    site_info=site_info,
                    site_params={
                        **hdf5_site_params,
                        "augmentations": TRAIN_AUGMENTATIONS,
                        "subset": {"eval_ratio": eval_ratio, "use_eval_split": False},
                    },
                    prefix="train",
                    dataset_hyper_params=generic_site_params,
                    segm_class_count=SEGMENTATION_CLASS_COUNT,
                )
            )
            valid_parts.append(
                fbp_data_module.FBPDataModule.create_parser(
                    site_info=site_info,
                    site_params={
                        **hdf5_site_params,
                        "subset": {"eval_ratio": eval_ratio, "use_eval_split": True},
                    },
                    prefix="valid",
                    dataset_hyper_params=generic_site_params,
                    segm_class_count=SEGMENTATION_CLASS_COUNT,
                )
            )
    else:
        raise ValueError(f"Unknown backend={backend!r}; use 'npz' or 'hdf5'")

    train_parser = _concat_or_single(train_parts)
    valid_parser = _concat_or_single(valid_parts)

    for site_name, tr, va in zip(site_names, train_parts, valid_parts):
        logger.info("  %s: train=%d valid=%d", site_name, len(tr), len(va))
    logger.info(
        "Total train gathers: %d | Valid gathers: %d",
        len(train_parser),
        len(valid_parser),
    )

    train_keys = {_gather_key(train_parser.get_meta_gather(i)) for i in range(len(train_parser))}
    valid_keys = {_gather_key(valid_parser.get_meta_gather(i)) for i in range(len(valid_parser))}
    assert not (train_keys & valid_keys), "train/valid gather keys overlap"
    logger.info("Train/valid gather keys are disjoint.")
    return train_parser, valid_parser


def build_loaders(
    train_parser,
    valid_parser,
    batch_size: int,
    num_workers: int,
):
    import hardpicks.data.fbp.data_module as fbp_data_module

    collate_fn = functools.partial(
        fbp_data_module.fbp_batch_collate,
        pad_to_nearest_pow2=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_parser,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_parser,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return str(value.detach().cpu().tolist())
        value = value.detach().cpu().item()
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


class ProgressMetricsCallback(pl.Callback):
    """Print train loss on a step interval and full metrics after each validation epoch."""

    def __init__(self, print_every_n_steps: int = 50):
        super().__init__()
        self.print_every_n_steps = max(0, int(print_every_n_steps))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.print_every_n_steps <= 0:
            return
        step = int(trainer.global_step)
        if step == 0 or step % self.print_every_n_steps != 0:
            return
        metrics = trainer.callback_metrics
        loss = metrics.get("train/loss")
        lr = metrics.get("train/learning_rate")
        parts = [
            f"[step {step:6d}]",
            f"epoch={trainer.current_epoch}",
            f"train/loss={_fmt_metric(loss)}",
        ]
        if lr is not None:
            parts.append(f"lr={_fmt_metric(lr)}")
        print(" | ".join(parts), flush=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip Lightning's sanity-check validation before epoch 0 training.
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        keys = sorted(
            k
            for k in metrics.keys()
            if isinstance(k, str) and (k.startswith("train/") or k.startswith("valid/"))
        )
        print("-" * 72, flush=True)
        print(
            f"Epoch {trainer.current_epoch} — intermediate metrics "
            f"(step {trainer.global_step})",
            flush=True,
        )
        for key in keys:
            print(f"  {key:32s} {_fmt_metric(metrics[key])}", flush=True)
        print("-" * 72, flush=True)


def _find_metrics_csv(csv_root: Path) -> Path:
    candidates = sorted(csv_root.rglob("metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv under {csv_root}")
    return candidates[-1]


def _epoch_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse step-wise CSVLogger rows into one row per epoch."""
    if "epoch" not in df.columns:
        raise ValueError("metrics.csv has no epoch column")
    rows = []
    for epoch, g in df.groupby("epoch", sort=True):
        row = {"epoch": int(epoch)}
        for col in g.columns:
            if col in {"epoch", "step"}:
                continue
            vals = g[col].dropna()
            if len(vals):
                row[col] = float(vals.iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def print_training_summary(
    output_root: Path,
    csv_dir: Path,
    site_label: str,
    max_epochs: int,
    n_train: int,
    n_valid: int,
    batch_size: int,
    best_path: Optional[Path],
    best_score: Any,
) -> pd.DataFrame:
    metrics_csv = _find_metrics_csv(csv_dir)
    raw_metrics = pd.read_csv(metrics_csv)
    epoch_df = _epoch_table(raw_metrics)
    epoch_csv = output_root / "epoch_metrics.csv"
    epoch_df.to_csv(epoch_csv, index=False)

    print("=" * 72)
    print(f"TRAINING SUMMARY — sites={site_label}  epochs={max_epochs}")
    print("=" * 72)
    print(f"Output dir     : {output_root}")
    print(f"Train gathers  : {n_train}")
    print(f"Valid gathers  : {n_valid}")
    print(f"Batch size     : {batch_size}")
    print(f"Best checkpoint: {best_path}")
    print(f"Best {MONITOR_METRIC}: {best_score}")
    print(f"Metrics CSV    : {metrics_csv}")
    print(f"Epoch CSV      : {epoch_csv}")
    print()

    display_cols = [
        c
        for c in epoch_df.columns
        if c == "epoch" or c.startswith("train") or c.startswith("valid")
    ]
    summary_table = epoch_df[display_cols].copy()
    print("Per-epoch metrics:")
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(summary_table.round(4).to_string(index=False))

    if len(epoch_df):
        last = epoch_df.iloc[-1]
        print("\nFinal epoch:")
        for c in display_cols:
            if c == "epoch":
                continue
            if pd.notna(last.get(c)):
                print(f"  {c:30s} {last[c]:.6g}")

        print("\nBest validation values across epochs:")
        for col in [c for c in epoch_df.columns if c.startswith("valid/")]:
            series = epoch_df[col].dropna()
            if series.empty:
                continue
            lower_better = any(
                k in col.lower() for k in ("error", "loss", "mae", "mse", "rmse", "bias")
            )
            if "bias" in col.lower() and "abs" not in col.lower():
                idx = series.abs().idxmin()
                tag = "closest-to-0"
            elif lower_better:
                idx = series.idxmin()
                tag = "min"
            else:
                idx = series.idxmax()
                tag = "max"
            ep = int(epoch_df.loc[idx, "epoch"])
            print(f"  {col:30s} {series.loc[idx]:.6g}  ({tag} @ epoch {ep})")

    return epoch_df


def save_metric_curves(epoch_df: pd.DataFrame, output_root: Path, site_label: str) -> Path:
    plot_specs: List[tuple] = []
    train_loss_cols = [c for c in epoch_df.columns if "loss" in c.lower() and c.startswith("train")]
    valid_loss_cols = [c for c in epoch_df.columns if "loss" in c.lower() and c.startswith("valid")]
    if train_loss_cols or valid_loss_cols:
        plot_specs.append(("Loss", train_loss_cols + valid_loss_cols))

    hit_cols = [c for c in epoch_df.columns if "HitRate" in c]
    if hit_cols:
        plot_specs.append(("Hit rate", hit_cols))

    err_cols = [
        c
        for c in epoch_df.columns
        if any(k in c for k in ("MeanAbsoluteError", "MeanBiasError", "MAE", "MBE"))
    ]
    if err_cols:
        plot_specs.append(("Pick error", err_cols))

    if not plot_specs:
        cols = [c for c in epoch_df.columns if c.startswith(("train", "valid"))]
        plot_specs.append(("Metrics", cols))

    n = len(plot_specs)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (title, cols) in zip(axes, plot_specs):
        for col in cols:
            s = epoch_df[["epoch", col]].dropna()
            if s.empty:
                continue
            ax.plot(s["epoch"], s[col], marker="o", label=col)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(f"{site_label} — training / validation curves", y=1.01)
    fig.tight_layout()
    curves_path = output_root / "train_valid_curves.png"
    fig.savefig(curves_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return curves_path


def make_trainer(
    *,
    tbx_logger,
    csv_logger,
    callbacks: Iterable[pl.Callback],
    max_epochs: int,
    log_every_n_steps: int,
) -> pl.Trainer:
    trainer_kwargs = dict(
        logger=[tbx_logger, csv_logger],
        callbacks=list(callbacks),
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
    )
    pl_major = int(str(pl.__version__).split(".", 1)[0])
    if pl_major >= 2:
        return pl.Trainer(
            **trainer_kwargs,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else "auto",
        )
    return pl.Trainer(
        **trainer_kwargs,
        gpus=int(bool(torch.cuda.device_count())),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        list_models()
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    site_names = [s.strip() for s in args.sites.split(",") if s.strip()]
    if not site_names:
        raise SystemExit("--sites must list at least one site")
    site_label = "_".join(s.lower() for s in site_names)

    model_config, model_label = build_model_config(
        model=args.model,
        max_epochs=args.epochs,
        lr=args.lr,
        model_config_path=args.model_config,
        encoder_weights=args.encoder_weights,
    )
    model_slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in model_label.lower())

    npz_root = args.npz_root or (args.data_dir / "npz")
    output_root = (
        args.output_dir
        or (REPO_ROOT / "output" / f"train_{site_label}_{model_slug}")
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("PL compat:", ensure_hardpicks_lightning_compat())
    if not hardpicks_available():
        raise SystemExit("hardpicks (+ torch) required — run setup_lightning.sh / install requirements")

    import hardpicks
    import models.fbp.unet as fbp_unet

    pl.seed_everything(args.seed, workers=True)

    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| pl", pl.__version__)
    print("hardpicks", hardpicks.__file__)
    print("FBPUNet module:", fbp_unet.__file__)
    print("Experiment dir:", output_root)
    print("Sites:", site_names)
    print("Model:", model_label, "| encoder:", model_config.get("unet_encoder_type"))
    print("encoder_weights:", model_config.get("encoder_weights"))
    print("DATA_BACKEND:", args.backend, "| NPZ_ROOT:", npz_root)

    config_out = output_root / "model_config.yaml"
    with config_out.open("w") as f:
        yaml.safe_dump(model_config, f, sort_keys=False, default_flow_style=False)
    print("Wrote", config_out)

    tbx_dir = output_root / "tensorboard"
    csv_dir = output_root / "csv_logs"
    tbx_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    tbx_logger = pl.loggers.TensorBoardLogger(
        save_dir=str(tbx_dir), name="default", default_hp_metric=False
    )
    csv_logger = pl.loggers.CSVLogger(save_dir=str(csv_dir), name="metrics")
    print("TensorBoard:", f"tensorboard --logdir {tbx_dir}")

    train_parser, valid_parser = build_parsers(
        site_names=site_names,
        backend=args.backend,
        data_dir=args.data_dir,
        npz_root=npz_root,
        eval_ratio=args.eval_ratio,
    )
    train_loader, valid_loader = build_loaders(
        train_parser,
        valid_parser,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")

    model = fbp_unet.FBPUNet(model_config)
    setattr(model, "_tbx_logger", tbx_logger)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FBPUNet[{model_label}] ready: {n_params / 1e6:.2f}M trainable parameters")

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=str(output_root),
        filename="best-{epoch:03d}-{step:06d}",
        monitor=MONITOR_METRIC,
        mode="max",
        save_top_k=1,
    )
    progress_cb = ProgressMetricsCallback(print_every_n_steps=args.print_every_n_steps)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = make_trainer(
        tbx_logger=tbx_logger,
        csv_logger=csv_logger,
        callbacks=[checkpoint_cb, progress_cb, lr_monitor],
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
    )

    print("Device:", "GPU" if torch.cuda.is_available() else "CPU")
    _batch = next(iter(train_loader))
    print("batch keys:", sorted(_batch.keys()))
    print("samples", tuple(_batch["samples"].shape), _batch["samples"].dtype)

    print(f"Training for {args.epochs} epochs…")
    trainer.fit(model, train_loader, valid_loader)

    best_path = Path(checkpoint_cb.best_model_path).resolve() if checkpoint_cb.best_model_path else None
    print("Best checkpoint:", best_path)
    print("Best score:", checkpoint_cb.best_model_score)

    epoch_df = print_training_summary(
        output_root=output_root,
        csv_dir=csv_dir,
        site_label=site_label,
        max_epochs=args.epochs,
        n_train=len(train_parser),
        n_valid=len(valid_parser),
        batch_size=args.batch_size,
        best_path=best_path,
        best_score=checkpoint_cb.best_model_score,
    )
    curves_path = save_metric_curves(epoch_df, output_root, site_label)
    print("Saved", curves_path)

    if not args.no_final_validate and best_path and best_path.is_file():
        best_model = fbp_unet.FBPUNet.load_from_checkpoint(str(best_path))
        setattr(best_model, "_tbx_logger", tbx_logger)
        try:
            val_out = trainer.validate(best_model, dataloaders=valid_loader)
        except TypeError:
            val_out = trainer.validate(best_model, val_dataloaders=valid_loader)
        print("\nValidation with best checkpoint:")
        if isinstance(val_out, list) and val_out:
            for k, v in sorted(val_out[0].items()):
                print(f"  {k:30s} {v}")
    elif args.no_final_validate:
        print("Skipped final validate (--no-final-validate).")
    else:
        print("No best checkpoint on disk; skipped final validate().")

    print("\nDone.")
    print(f"TensorBoard: tensorboard --logdir {tbx_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
