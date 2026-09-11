#!/usr/bin/env python3
"""Train local FBPUNet (cloned from hardpicks) on NPZ or HDF5 gathers.

Reports to TensorBoard + CSV, writes a living report under ``report/`` (intermediate
train loss, per-epoch validation, final metrics, and best-checkpoint path), and
saves the best checkpoint on ``valid/HitRate1px``.

The trainer class lives in ``models.fbp.unet.FBPUNet``. Different “models” are
encoder/architecture presets (ResNet18, EfficientNet-B0, …) or a YAML/JSON
hyperparameter file. Optional SMP backbone pretraining via ``--encoder-weights``.

Example::

    conda activate seismic_activity
    python train/fbp_train.py --sites Brunswick --model resnet18
    python train/fbp_train.py --sites Brunswick --model efficientnet-b0 \\
        --encoder-weights imagenet
    python train/fbp_train.py --sites Brunswick --model-config my_model.yaml
    python train/fbp_train.py --fold A --model resnet18
    python train/fbp_train.py --fold A --patience 0  # train all --epochs, no early stop
    python train/fbp_train.py --config configs/train.yaml --fold A
    python train/fbp_train.py --list-folds
    python train/fbp_train.py --fold A --ckpt output/train_foldA_resnet18/best-epoch=013-step=015232.ckpt
    python train/fbp_train.py --list-folds
    # live report: report/train_foldA_resnet18_YYYYMMDD_HHMMSS/report.md

    # multi-GPU (batch size is per GPU). Prefer torchrun for DDP:
    torchrun --nproc_per_node=4 train/fbp_train.py --fold A --devices 4
    python train/fbp_train.py --fold A --devices 2 --strategy ddp_spawn
    python train/fbp_train.py --fold A --devices 0,1 --strategy dp

    tensorboard --logdir output/train_brunswick_resnet18/tensorboard
"""

from __future__ import annotations

import argparse
import copy
import csv
import functools
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seismic_utils.dataset import DEFAULT_DATA_DIR
from seismic_utils.hardpicks_bridge import hardpicks_available, resolve_hardpicks_site_info
from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat
from seismic_utils.npz_parser import create_npz_parser
from seismic_utils.predict import resolve_checkpoint

logger = logging.getLogger("fbp_train")

MONITOR_METRIC = "valid/HitRate1px"
SEGMENTATION_CLASS_COUNT = 1
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"

# argparse dest names filled from configs/train.yaml (not model-config).
_RECIPE_TRAINER_KEYS = (
    "seed",
    "backend",
    "eval_ratio",
    "epochs",
    "batch_size",
    "patience",
    "precision",
    "num_workers",
    "log_every_n_steps",
    "print_every_n_steps",
    "model",
)

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

# Cross-site folds from hardpicks ``data/fbp/folds/foldA.yaml`` … ``foldK.yaml``.
# Each named fold holds out whole sites (no intra-site ``eval_ratio`` split).
# A–D are 3 train / 1 valid on the four sites in this repo (Brunswick, Halfmile,
# Lalor, Sudbury). Fold A matches hardpicks exactly; B–D are the same rotation
# with Kevitsa/Matagami dropped (that 5th site is not in this dataset).
# F–K match the hardpicks YAMLs that already omit Kevitsa (2 train / 1 valid).
SITE_FOLDS: Dict[str, Dict[str, List[str]]] = {
    "A": {"train": ["Lalor", "Brunswick", "Sudbury"], "valid": ["Halfmile"]},
    "B": {"train": ["Lalor", "Brunswick", "Halfmile"], "valid": ["Sudbury"]},
    "C": {"train": ["Halfmile", "Lalor", "Sudbury"], "valid": ["Brunswick"]},
    "D": {"train": ["Sudbury", "Halfmile", "Brunswick"], "valid": ["Lalor"]},
    "F": {"train": ["Halfmile", "Brunswick"], "valid": ["Sudbury"]},
    "G": {"train": ["Brunswick", "Sudbury"], "valid": ["Halfmile"]},
    "H": {"train": ["Halfmile", "Lalor"], "valid": ["Brunswick"]},
    "I": {"train": ["Sudbury", "Halfmile"], "valid": ["Lalor"]},
    "J": {"train": ["Lalor", "Brunswick"], "valid": ["Sudbury"]},
    "K": {"train": ["Brunswick", "Sudbury"], "valid": ["Halfmile"]},
}

# Folds that exist in hardpicks but cannot run here (need Kevitsa/Matagami).
UNAVAILABLE_FOLDS: Dict[str, str] = {
    "E": "validates on Matagami/Kevitsa, which is not in this dataset",
}

# Allow hardpicks / SMP encoder spellings to resolve to the same preset.
MODEL_ALIASES: Dict[str, str] = {
    "timm-efficientnet-b0": "efficientnet-b0",
    "timm-efficientnet-b4": "efficientnet-b4",
    "fbpunet": "resnet18",
    "fbp-unet": "resnet18",
}

# Used only when the recipe omits ``augmentations`` (missing config file / old YAML).
DEFAULT_TRAIN_AUGMENTATIONS: List[Dict[str, Any]] = [
    {
        "type": "crop",
        "params": {
            "low_sample_count": 512,
            "high_sample_count": 1024,
            "max_crop_fraction": 0.333,
        },
    },
    {"type": "kill", "params": {"prob": 0.08}},
    {
        "type": "drop_and_pad",
        "params": {
            "target_trace_counts": [64, 128, 256, 512],
            "full_snap": True,
            "max_drop_ratio": 0.50,
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


def _argv_list(argv: Optional[Sequence[str]]) -> List[str]:
    if argv is None:
        return sys.argv[1:]
    return list(argv)


def _config_flag_explicit(argv: Sequence[str]) -> bool:
    return any(arg == "--config" or arg.startswith("--config=") for arg in argv)


def resolve_train_config_path(path: Path) -> Path:
    raw = Path(path).expanduser()
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((REPO_ROOT / raw).resolve())
    seen = set()
    uniq: List[Path] = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        uniq.append(cand)
        if cand.is_file():
            return cand
    return uniq[0] if uniq else raw.resolve()


def load_train_recipe(path: Path, *, required: bool) -> tuple[Dict[str, Any], Path]:
    resolved = resolve_train_config_path(path)
    if not resolved.is_file():
        if required:
            raise SystemExit(f"train config not found: {path}")
        logger.warning("train config not found (%s); using built-in argparse defaults", path)
        return {}, resolved
    data = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"train config must be a mapping: {resolved}")
    return data, resolved


def _trainer_defaults_from_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    return {key: recipe[key] for key in _RECIPE_TRAINER_KEYS if recipe.get(key) is not None}


def resolve_train_augmentations(recipe: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Train-only augs from the recipe YAML.

    Missing key → built-in default. ``null`` or ``[]`` disables augs. A mapping
    (hardpicks-style named ops) is accepted and converted to a list.
    """
    if "augmentations" not in recipe:
        return copy.deepcopy(DEFAULT_TRAIN_AUGMENTATIONS)
    raw = recipe["augmentations"]
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        raise SystemExit("train config 'augmentations' must be a list, mapping, or null")
    ops: List[Dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or not item.get("type"):
            raise SystemExit(
                f"train config augmentations[{i}] must be a mapping with a 'type' key"
            )
        ops.append(copy.deepcopy(item))
    return ops


def _aug_type_names(augmentations: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(op.get("type", "?")) for op in augmentations]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    argv_list = _argv_list(argv)
    preset_names = ", ".join(sorted(MODEL_PRESETS))
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_TRAIN_CONFIG,
        help="Training recipe YAML (loop, split, model, loss/LR, augmentations). CLI flags override it.",
    )
    pre_args, _ = pre.parse_known_args(argv_list)
    recipe, config_path = load_train_recipe(
        pre_args.config, required=_config_flag_explicit(argv_list)
    )
    p = argparse.ArgumentParser(
        description=(
            "Train FBPUNet first-break picker (NPZ or HDF5). "
            "Use --sites for an intra-site split, or --fold A–K for hardpicks whole-site holdout. "
            "Recipe defaults: configs/train.yaml (--config)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[pre],
    )
    p.add_argument(
        "--sites",
        default=None,
        help=(
            "Comma-separated site names (Brunswick | Halfmile | Lalor | Sudbury). "
            "Each site is split with --eval-ratio. Mutually exclusive with --fold. "
            "Default when neither --sites nor --fold is set: Brunswick,Halfmile."
        ),
    )
    p.add_argument(
        "--fold",
        default=None,
        metavar="ID",
        help=(
            "Hardpicks cross-site fold (A–D: 3 train / 1 valid; F–K: 2 train / 1 valid). "
            "Accepts A, foldA, fold_a. Whole sites are held out (ignores --eval-ratio). "
            "Mutually exclusive with --sites."
        ),
    )
    p.add_argument(
        "--list-folds",
        action="store_true",
        help="Print hardpicks-style site folds and exit.",
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
    p.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Max training epochs (early stopping may halt sooner).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-GPU (per-device) batch size. Global batch is this times the GPU count.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early-stop patience on valid/HitRate1px (0 disables early stopping).",
    )
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--devices",
        default="auto",
        help=(
            "Lightning devices: auto/-1 = all visible GPUs (or CPU), an integer count "
            "(e.g. 4), or comma-separated GPU ids (e.g. 0,1). Use CUDA_VISIBLE_DEVICES "
            "to limit which GPUs are visible."
        ),
    )
    p.add_argument(
        "--accelerator",
        default="auto",
        choices=("auto", "gpu", "cpu"),
        help="Training accelerator.",
    )
    p.add_argument(
        "--strategy",
        default="auto",
        help=(
            "Distributed strategy when using >1 GPU: auto (DDP), ddp, ddp_spawn, dp. "
            "Prefer torchrun + ddp; ddp_spawn/dp work from a plain python launch."
        ),
    )
    p.add_argument(
        "--precision",
        default="32",
        help="Floating-point precision (32, 16-mixed, bf16-mixed).",
    )
    p.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help="DDP find_unused_parameters=True (slower; use if unused-parameter errors).",
    )
    p.add_argument(
        "--eval-ratio",
        type=float,
        default=0.15,
        help="Per-site validation fraction (ignored when --fold is set).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Experiment directory (default: output/train_<sites-or-fold>_<model>).",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Training report root (default: <repo>/report/<run-name>_<YYYYMMDD_HHMMSS>/).",
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
    p.add_argument(
        "--lr-step",
        type=int,
        default=None,
        choices=(5, 10, 20),
        help="StepLR step_size in epochs (default: 10). 20 with --epochs 20 is effectively constant LR.",
    )
    p.add_argument(
        "--loss",
        default=None,
        choices=("crossentropy", "dice"),
        help="Segmentation loss (default: crossentropy, or --model-config).",
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Resume Lightning training from this .ckpt (weights, optimizer, epoch).",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory of best*.ckpt; resumes from the newest (used if --ckpt is omitted).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-final-validate",
        action="store_true",
        help="Skip re-validation with the best checkpoint after fit.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(**_trainer_defaults_from_recipe(recipe))
    args = p.parse_args(argv_list)
    args.train_recipe = recipe
    args.train_config_path = config_path
    args.train_augmentations = resolve_train_augmentations(recipe)
    return args


def resolve_model_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if key in MODEL_PRESETS:
        return key
    # Accept SMP / hardpicks encoder ids as custom model labels.
    return name.strip()


def normalize_fold_id(name: str) -> str:
    """Accept ``A``, ``foldA``, ``fold_a``, ``Fold A`` → ``A``."""
    key = name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key.startswith("fold"):
        key = key[4:]
    return key.upper()


def resolve_fold(name: str) -> tuple[str, List[str], List[str]]:
    """Return ``(fold_id, train_sites, valid_sites)`` for a hardpicks fold letter."""
    fold_id = normalize_fold_id(name)
    if fold_id in UNAVAILABLE_FOLDS:
        raise SystemExit(
            f"Fold {fold_id} is defined in hardpicks but {UNAVAILABLE_FOLDS[fold_id]}. "
            f"Use --list-folds for splits that run with Brunswick/Halfmile/Lalor/Sudbury."
        )
    spec = SITE_FOLDS.get(fold_id)
    if spec is None:
        known = ", ".join(sorted(SITE_FOLDS))
        raise SystemExit(f"Unknown fold {name!r}. Known folds: {known}. Try --list-folds.")
    train_sites = list(spec["train"])
    valid_sites = list(spec["valid"])
    overlap = set(train_sites) & set(valid_sites)
    if overlap:
        raise SystemExit(f"Fold {fold_id} train/valid sites overlap: {sorted(overlap)}")
    return fold_id, train_sites, valid_sites


def list_folds() -> None:
    print("Hardpicks-style site folds (--fold). Whole sites are held out.\n")
    print("3 train / 1 valid (leave-one-site-out on Brunswick, Halfmile, Lalor, Sudbury):")
    for fold_id in "ABCD":
        spec = SITE_FOLDS[fold_id]
        train = ", ".join(spec["train"])
        valid = ", ".join(spec["valid"])
        print(f"  {fold_id}    train: {train}")
        print(f"       valid: {valid}")
    print("\nHardpicks YAMLs without Kevitsa (2 train / 1 valid):")
    for fold_id in "FGHIJK":
        spec = SITE_FOLDS[fold_id]
        train = ", ".join(spec["train"])
        valid = ", ".join(spec["valid"])
        print(f"  {fold_id}    train: {train}")
        print(f"       valid: {valid}")
    print("\nUnavailable here:")
    for fold_id, reason in sorted(UNAVAILABLE_FOLDS.items()):
        print(f"  {fold_id}    {reason}")
    print("\nExample: python train/fbp_train.py --fold A --epochs 20")


def resolve_train_valid_sites(args: argparse.Namespace) -> tuple[str, List[str], List[str], Optional[float]]:
    """Return ``(site_label, train_sites, valid_sites, eval_ratio_or_none)``."""
    if args.fold and args.sites:
        raise SystemExit("Use either --fold or --sites, not both.")
    if args.fold:
        fold_id, train_sites, valid_sites = resolve_fold(args.fold)
        return f"fold{fold_id}", train_sites, valid_sites, None
    sites_arg = args.sites or "Brunswick,Halfmile"
    site_names = [s.strip() for s in sites_arg.split(",") if s.strip()]
    if not site_names:
        raise SystemExit("--sites must list at least one site")
    site_label = "_".join(s.lower() for s in site_names)
    return site_label, site_names, list(site_names), float(args.eval_ratio)


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
    loss_type: Optional[str] = None,
    lr_step: Optional[int] = None,
    recipe: Optional[Dict[str, Any]] = None,
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
    recipe = dict(recipe or {})
    if recipe.get("lr") is not None:
        preset_lr = float(recipe["lr"])
    recipe_loss = recipe.get("loss")
    recipe_lr_step = recipe.get("lr_step")
    recipe_encoder = recipe.get("encoder_weights")
    file_lr = file_overrides.pop("lr", None)
    if isinstance(file_lr, dict):
        file_lr = None  # ignore accidental nested structures
    # optimizer_params.lr in the file still wins later via deep merge below.

    recipe_encoder_w = None
    if recipe_encoder not in (None, "", "none", "null"):
        recipe_encoder_w = str(recipe_encoder)

    base = {
        "model_type": "FBPUNet",
        "unet_decoder_type": "vanilla",
        "decoder_attention_type": None,
        "segm_class_count": SEGMENTATION_CLASS_COUNT,
        "use_dist_offsets": True,
        "use_first_break_prior": False,
        "coordconv": False,
        "encoder_weights": recipe_encoder_w,
        "optimizer_type": "Adam",
        "optimizer_params": {"lr": preset_lr, "weight_decay": 1e-6},
        "scheduler_type": "StepLR",
        "scheduler_params": {
            "step_size": int(recipe_lr_step) if recipe_lr_step is not None else 10,
            "gamma": 0.1,
        },
        "update_scheduler_at_epochs": True,
        "loss_type": str(recipe_loss) if recipe_loss else "crossentropy",
        "loss_params": {},
        "use_full_metrics_during_training": False,
        "eval_type": "FBPEvaluator",
        "segm_first_break_prob_threshold": 0.0,
        "eval_metrics": [
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 1}},
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 3}},
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 5}},
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 7}},
            {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 9}},
            {"metric_type": "MeanBiasError"},
            {"metric_type": "MeanAbsoluteError"},
            {"metric_type": "RootMeanSquaredError"},
            {"metric_type": "GatherCoverage"},
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

    if loss_type is not None:
        base["loss_type"] = str(loss_type)
    if lr_step is not None:
        sched = dict(base.get("scheduler_params") or {})
        sched["step_size"] = int(lr_step)
        base["scheduler_params"] = sched
    # Binary first-break vs not (ternary is not supported in this trainer).
    base["segm_class_count"] = SEGMENTATION_CLASS_COUNT

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


def _site_params(
    *,
    augment: bool,
    eval_ratio: Optional[float],
    use_eval_split: bool,
    augmentations: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = dict(COMMON_SITE_PARAMS)
    if augment:
        ops = (
            list(augmentations)
            if augmentations is not None
            else DEFAULT_TRAIN_AUGMENTATIONS
        )
        if ops:
            # ShotLineGatherPreprocessor mutates each aug dict (replaces type str
            # with a callable). Copy so the 2nd+ training site does not see a
            # spent config.
            params["augmentations"] = copy.deepcopy(ops)
    if eval_ratio is not None:
        params["subset"] = {"eval_ratio": eval_ratio, "use_eval_split": use_eval_split}
    return params


def build_split_parser(
    site_names: Sequence[str],
    *,
    prefix: str,
    backend: str,
    data_dir: Path,
    npz_root: Path,
    eval_ratio: Optional[float],
    augment: bool,
    use_eval_split: bool,
    segm_class_count: int = SEGMENTATION_CLASS_COUNT,
    augmentations: Optional[Sequence[Dict[str, Any]]] = None,
):
    """Build a concatenated parser for one split (train or valid)."""
    import hardpicks
    import hardpicks.data.fbp.data_module as fbp_data_module

    parts: list = []
    backend = backend.strip().lower()
    if not site_names:
        raise ValueError(f"no sites provided for {prefix} split")

    if backend == "npz":
        for site_name in site_names:
            logger.info("NPZ parser (%s): %s", prefix, npz_root / site_name)
            parts.append(
                create_npz_parser(
                    site_name,
                    npz_root=npz_root,
                    prefix=prefix,
                    site_params=_site_params(
                        augment=augment,
                        eval_ratio=eval_ratio,
                        use_eval_split=use_eval_split,
                        augmentations=augmentations,
                    ),
                    segm_class_count=segm_class_count,
                )
            )
    elif backend == "hdf5":
        rejected = Path(hardpicks.FBP_BAD_GATHERS_DIR) / "bad-gather-ids_combined.yaml"
        if not rejected.is_file():
            rejected = None
            logger.warning("bad-gather YAML not found; continuing without reject list")
        hdf5_site_params = {
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
            logger.info("HDF5 parser (%s): %s", prefix, site_name)
            for k, v in site_info.items():
                logger.info("  %s: %s", k, v)
            parts.append(
                fbp_data_module.FBPDataModule.create_parser(
                    site_info=site_info,
                    site_params={
                        **hdf5_site_params,
                        **_site_params(
                            augment=augment,
                            eval_ratio=eval_ratio,
                            use_eval_split=use_eval_split,
                            augmentations=augmentations,
                        ),
                    },
                    prefix=prefix,
                    dataset_hyper_params=generic_site_params,
                    segm_class_count=segm_class_count,
                )
            )
    else:
        raise ValueError(f"Unknown backend={backend!r}; use 'npz' or 'hdf5'")

    for site_name, parser in zip(site_names, parts):
        logger.info("  %s %s: %d gathers", prefix, site_name, len(parser))
    return _concat_or_single(parts)


def build_parsers(
    train_site_names: Sequence[str],
    valid_site_names: Sequence[str],
    backend: str,
    data_dir: Path,
    npz_root: Path,
    eval_ratio: Optional[float],
    segm_class_count: int = SEGMENTATION_CLASS_COUNT,
    augmentations: Optional[Sequence[Dict[str, Any]]] = None,
):
    train_parser = build_split_parser(
        train_site_names,
        prefix="train",
        backend=backend,
        data_dir=data_dir,
        npz_root=npz_root,
        eval_ratio=eval_ratio,
        augment=True,
        use_eval_split=False,
        segm_class_count=segm_class_count,
        augmentations=augmentations,
    )
    valid_parser = build_split_parser(
        valid_site_names,
        prefix="valid",
        backend=backend,
        data_dir=data_dir,
        npz_root=npz_root,
        eval_ratio=eval_ratio,
        augment=False,
        use_eval_split=True,
        segm_class_count=segm_class_count,
    )
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
    *,
    pin_memory: bool = False,
    drop_last_train: bool = False,
    shuffle_train: bool = True,
):
    import hardpicks.data.fbp.data_module as fbp_data_module

    worker_kwargs: Dict[str, Any] = {}
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = True
        worker_kwargs["prefetch_factor"] = 2
    collate_fn = functools.partial(
        fbp_data_module.fbp_batch_collate,
        pad_to_nearest_pow2=True,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_parser,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        **worker_kwargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_parser,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **worker_kwargs,
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


def _metric_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _snapshot_metrics(trainer) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        if not isinstance(key, str):
            continue
        parsed = _metric_float(value)
        if parsed is not None:
            out[key] = parsed
    return out


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _md_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows yet._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if value is None or value == "":
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


class TrainingReport:
    """Living training report under ``<repo>/report/<run>/``."""

    _TRAIN_FIELDS = ("time", "step", "epoch", "train/loss", "lr")

    def __init__(self, report_dir: Path, meta: Dict[str, Any]):
        self.root = Path(report_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.meta.setdefault("started_at", _utc_now())
        self.meta["status"] = "running"
        self.train_csv = self.root / "train_steps.csv"
        self.valid_jsonl = self.root / "valid_epochs.jsonl"
        self.report_md = self.root / "report.md"
        self.final_json = self.root / "final.json"
        self.train_rows: List[Dict[str, Any]] = []
        self.valid_rows: List[Dict[str, Any]] = []
        self.best_checkpoint: Optional[str] = None
        self.best_score: Optional[float] = None
        self.final_valid: Optional[Dict[str, Any]] = None
        self.curves_path: Optional[str] = None
        with self.train_csv.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=list(self._TRAIN_FIELDS)).writeheader()
        self.valid_jsonl.write_text("")
        self._write_markdown()

    def log_train_step(self, *, step: int, epoch: int, loss: Any, lr: Any = None) -> None:
        row = {
            "time": _utc_now(),
            "step": int(step),
            "epoch": int(epoch),
            "train/loss": _metric_float(loss),
            "lr": _metric_float(lr),
        }
        self.train_rows.append(row)
        with self.train_csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(self._TRAIN_FIELDS)).writerow(row)
        self._write_markdown()

    def log_validation(
        self,
        *,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        best_checkpoint: Optional[Path] = None,
        best_score: Any = None,
    ) -> None:
        row: Dict[str, Any] = {
            "time": _utc_now(),
            "step": int(step),
            "epoch": int(epoch),
            **{k: v for k, v in metrics.items() if k.startswith(("train/", "valid/"))},
        }
        self.valid_rows.append(row)
        with self.valid_jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if best_checkpoint:
            self.best_checkpoint = str(Path(best_checkpoint).resolve())
        score = _metric_float(best_score)
        if score is not None:
            self.best_score = score
        elif MONITOR_METRIC in metrics:
            if self.best_score is None or metrics[MONITOR_METRIC] > self.best_score:
                self.best_score = metrics[MONITOR_METRIC]
        self._write_markdown()

    def finalize(
        self,
        *,
        best_path: Optional[Path],
        best_score: Any,
        epoch_df: Optional[pd.DataFrame] = None,
        extra_valid: Optional[Dict[str, Any]] = None,
        curves_path: Optional[Path] = None,
    ) -> Path:
        self.meta["status"] = "finished"
        self.meta["finished_at"] = _utc_now()
        if best_path:
            self.best_checkpoint = str(Path(best_path).resolve())
        score = _metric_float(best_score)
        if score is not None:
            self.best_score = score
        if extra_valid:
            self.final_valid = {
                k: _metric_float(v) if _metric_float(v) is not None else v
                for k, v in extra_valid.items()
            }
        if curves_path and Path(curves_path).is_file():
            dest = self.root / Path(curves_path).name
            shutil.copy2(curves_path, dest)
            self.curves_path = str(dest)
        if epoch_df is not None and len(epoch_df):
            epoch_csv = self.root / "epoch_metrics.csv"
            epoch_df.to_csv(epoch_csv, index=False)
        payload = {
            **self.meta,
            "best_checkpoint": self.best_checkpoint,
            "best_metric": MONITOR_METRIC,
            "best_score": self.best_score,
            "final_validation": self.final_valid,
            "curves": self.curves_path,
            "report_markdown": str(self.report_md),
        }
        _atomic_write_text(self.final_json, json.dumps(payload, indent=2, default=str) + "\n")
        self._write_markdown()
        return self.report_md

    def _write_markdown(self) -> None:
        meta = self.meta
        train_preview = self.train_rows[-30:]
        valid_cols = ["epoch", "step"]
        extra_cols: List[str] = []
        for row in self.valid_rows:
            for key in row:
                if key in extra_cols or key in {"time", "epoch", "step"}:
                    continue
                if key.startswith("valid/") or key == "train/loss":
                    extra_cols.append(key)
        extra_cols.sort()
        if "train/loss" in extra_cols:
            extra_cols.remove("train/loss")
            extra_cols.insert(0, "train/loss")
        if MONITOR_METRIC in extra_cols:
            extra_cols.remove(MONITOR_METRIC)
            extra_cols.insert(0, MONITOR_METRIC)
        valid_cols.extend(extra_cols)

        lines = [
            f"# FBPUNet training report — {meta.get('run_name', '')}",
            "",
            f"**Status:** {meta.get('status', 'running')}",
            f"**Started:** {meta.get('started_at', '')}",
        ]
        if meta.get("finished_at"):
            lines.append(f"**Finished:** {meta['finished_at']}")
        lines.extend(
            [
                f"**Model:** {meta.get('model', '')}",
                f"**Encoder:** {meta.get('encoder', '')}",
                f"**Train sites:** {', '.join(meta.get('train_sites') or [])}",
                f"**Valid sites:** {', '.join(meta.get('valid_sites') or [])}",
            ]
        )
        if meta.get("fold"):
            lines.append(f"**Fold:** {meta['fold']}")
        if meta.get("eval_ratio") is not None:
            lines.append(f"**Eval ratio:** {meta['eval_ratio']}")
        if meta.get("resume_ckpt"):
            lines.append(f"**Resume ckpt:** `{meta['resume_ckpt']}`")
        if meta.get("n_train") is not None:
            lines.append(f"**Train gathers:** {meta['n_train']}")
        if meta.get("n_valid") is not None:
            lines.append(f"**Valid gathers:** {meta['n_valid']}")
        lines.extend(
            [
                f"**Epochs:** {meta.get('epochs', '')}",
                f"**Early-stop patience:** {meta.get('patience', '')}",
                f"**Batch size (per device):** {meta.get('batch_size', '')}",
                f"**Loss:** {meta.get('loss', '')}",
                f"**LR step:** {meta.get('lr_step', '')}",
                f"**Augmentations:** {', '.join(meta.get('augmentations') or []) or 'none'}",
                f"**Devices:** {meta.get('num_devices', meta.get('devices', ''))}",
                f"**Strategy:** {meta.get('strategy', '')}",
                f"**Backend:** {meta.get('backend', '')}",
                f"**Train config:** `{meta.get('train_config', '')}`",
                f"**Experiment dir:** `{meta.get('output_dir', '')}`",
                "",
                "## Best checkpoint",
                "",
            ]
        )
        if self.best_checkpoint:
            lines.append(f"- Path: `{self.best_checkpoint}`")
            lines.append(f"- {MONITOR_METRIC}: {_fmt_metric(self.best_score)}")
        else:
            lines.append("_Not saved yet._")
        lines.extend(
            [
                "",
                "## Intermediate training loss",
                "",
                f"Latest {len(train_preview)} of {len(self.train_rows)} logged steps "
                f"(full history: `{self.train_csv.name}`).",
                "",
                _md_table(train_preview, ["step", "epoch", "train/loss", "lr"]),
                "## Intermediate validation",
                "",
                f"{len(self.valid_rows)} epoch(s). Full records: `{self.valid_jsonl.name}`.",
                "",
                _md_table(self.valid_rows, valid_cols),
            ]
        )
        if self.final_valid:
            lines.extend(["## Final validation (best checkpoint)", ""])
            for key, value in sorted(self.final_valid.items()):
                lines.append(f"- `{key}`: {_fmt_metric(value)}")
            lines.append("")
        if self.curves_path:
            lines.extend(["## Curves", "", f"![train/valid curves]({Path(self.curves_path).name})", ""])
        _atomic_write_text(self.report_md, "\n".join(lines) + "\n")


class ProgressMetricsCallback(pl.Callback):
    """Print train loss on a step interval and full metrics after each validation epoch."""

    def __init__(
        self,
        print_every_n_steps: int = 50,
        report: Optional[TrainingReport] = None,
        checkpoint_cb: Optional[pl.callbacks.ModelCheckpoint] = None,
    ):
        super().__init__()
        self.print_every_n_steps = max(0, int(print_every_n_steps))
        self.report = report
        self.checkpoint_cb = checkpoint_cb
        self._report_train_every = self.print_every_n_steps if self.print_every_n_steps > 0 else 50

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        step = int(trainer.global_step)
        should_print = self.print_every_n_steps > 0 and step > 0 and step % self.print_every_n_steps == 0
        should_report = (
            self.report is not None
            and step > 0
            and step % self._report_train_every == 0
        )
        if not should_print and not should_report:
            return
        if getattr(trainer, "global_rank", 0) != 0:
            return
        metrics = trainer.callback_metrics
        loss = metrics.get("train/loss")
        lr = metrics.get("train/learning_rate")
        if should_print:
            parts = [
                f"[step {step:6d}]",
                f"epoch={trainer.current_epoch}",
                f"train/loss={_fmt_metric(loss)}",
            ]
            if lr is not None:
                parts.append(f"lr={_fmt_metric(lr)}")
            print(" | ".join(parts), flush=True)
        if should_report:
            self.report.log_train_step(
                step=step,
                epoch=int(trainer.current_epoch),
                loss=loss,
                lr=lr,
            )

    def on_validation_end(self, trainer, pl_module):
        # Skip Lightning's sanity-check validation before epoch 0 training.
        if trainer.sanity_checking:
            return
        if getattr(trainer, "global_rank", 0) != 0:
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
        if self.report is not None:
            best_path = None
            best_score = None
            if self.checkpoint_cb is not None:
                if self.checkpoint_cb.best_model_path:
                    best_path = Path(self.checkpoint_cb.best_model_path)
                best_score = self.checkpoint_cb.best_model_score
            self.report.log_validation(
                step=int(trainer.global_step),
                epoch=int(trainer.current_epoch),
                metrics=_snapshot_metrics(trainer),
                best_checkpoint=best_path,
                best_score=best_score,
            )


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


def env_is_rank_zero() -> bool:
    """True for the single-process case or the global rank-0 distributed worker."""
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        if key in os.environ:
            return int(os.environ[key]) == 0
    return True


def _pl_version() -> tuple[int, int]:
    parts = str(pl.__version__).split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
    return major, minor


def parse_devices(value: str) -> Union[int, List[int], str]:
    """Parse ``--devices`` into a Lightning ``devices`` argument."""
    raw = str(value).strip()
    key = raw.lower()
    if key in {"auto", "all", "-1"}:
        return "auto"
    if "," in raw or (raw.startswith("[") and raw.endswith("]")):
        ids = [int(part.strip()) for part in raw.strip("[]").split(",") if part.strip()]
        if not ids:
            raise ValueError(f"invalid --devices {value!r}")
        return ids
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(f"invalid --devices {value!r}") from exc
    if count < 0:
        return "auto"
    return count


def resolved_num_devices(devices: Union[int, List[int], str], *, use_gpu: bool) -> int:
    if not use_gpu:
        return 1
    if devices == "auto":
        return max(int(torch.cuda.device_count()), 1)
    if isinstance(devices, list):
        return len(devices)
    return max(int(devices), 1)


def map_precision(precision: str, pl_major: int) -> Any:
    key = str(precision).strip().lower()
    if key in {"32", "32-true", "fp32"}:
        return 32
    if key in {"16", "16-mixed", "fp16", "mixed"}:
        return 16 if pl_major < 2 else "16-mixed"
    if key in {"bf16", "bf16-mixed"}:
        return "bf16" if pl_major < 2 else "bf16-mixed"
    return precision


def resolve_strategy(
    strategy: str,
    *,
    num_devices: int,
    use_gpu: bool,
    find_unused_parameters: bool,
) -> Any:
    key = str(strategy).strip().lower()
    if key in {"none", "null", ""}:
        return None
    if key == "auto":
        if use_gpu and num_devices > 1:
            key = "ddp"
        else:
            return None
    if key in {"ddp", "ddp_spawn"} and find_unused_parameters:
        try:
            from pytorch_lightning.strategies import DDPStrategy

            if key == "ddp_spawn":
                from pytorch_lightning.strategies import DDPSpawnStrategy

                return DDPSpawnStrategy(find_unused_parameters=True)
            return DDPStrategy(find_unused_parameters=True)
        except Exception:
            return "ddp_find_unused_parameters_true" if key == "ddp" else key
    return key

def _load_fbpunet_from_checkpoint(ckpt_path: Path):
    """Reload a Lightning checkpoint without colliding with the parent MLflow run.

    ``hardpicks.utils.hp_utils.log_hp`` always calls ``mlflow.log_param``. Reconstructing
    a model from a checkpoint whose saved hparams differ from this run (e.g. resume a
    ``crossentropy`` ckpt then train with ``--loss dice``) raises
    ``UNIQUE constraint failed: params.key`` / ``Changing param values is not allowed``.
    """
    import models.fbp.unet as fbp_unet

    try:
        import mlflow
        from mlflow.exceptions import MlflowException
    except ImportError:
        return fbp_unet.FBPUNet.load_from_checkpoint(str(ckpt_path))

    orig = mlflow.log_param

    def _log_param_keep_existing(key, value, *args, **kwargs):
        try:
            return orig(key, value, *args, **kwargs)
        except MlflowException:
            logger.debug("mlflow: skip param overwrite %s=%r", key, value)
            return value

    mlflow.log_param = _log_param_keep_existing  # type: ignore[method-assign]
    try:
        return fbp_unet.FBPUNet.load_from_checkpoint(str(ckpt_path))
    finally:
        mlflow.log_param = orig

def make_trainer(
    *,
    tbx_logger,
    csv_logger,
    callbacks: Iterable[pl.Callback],
    max_epochs: int,
    log_every_n_steps: int,
    accelerator: str = "auto",
    devices: Union[int, List[int], str] = "auto",
    strategy: str = "auto",
    precision: str = "32",
    find_unused_parameters: bool = False,
) -> pl.Trainer:
    trainer_kwargs: Dict[str, Any] = dict(
        logger=[tbx_logger, csv_logger],
        callbacks=list(callbacks),
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        enable_progress_bar=True,
    )
    accel = str(accelerator).strip().lower()
    if accel == "cpu":
        use_gpu = False
    elif accel in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise SystemExit("--accelerator gpu requested but CUDA is not available")
        use_gpu = True
    else:
        use_gpu = bool(torch.cuda.is_available())

    n_devices = resolved_num_devices(devices, use_gpu=use_gpu)
    strategy_arg = resolve_strategy(
        strategy,
        num_devices=n_devices,
        use_gpu=use_gpu,
        find_unused_parameters=find_unused_parameters,
    )
    pl_major, _pl_minor = _pl_version()
    precision_arg = map_precision(precision, pl_major)
    sync_bn = bool(use_gpu and n_devices > 1 and str(strategy_arg or "ddp").startswith("ddp"))

    gpu_devices: Any
    if not use_gpu:
        gpu_devices = 1
    elif devices == "auto":
        gpu_devices = n_devices
    else:
        gpu_devices = devices

    variants: List[Dict[str, Any]] = []
    if pl_major >= 2:
        extra: Dict[str, Any] = {
            "accelerator": "gpu" if use_gpu else "cpu",
            "devices": gpu_devices if use_gpu else "auto",
            "precision": precision_arg,
            "sync_batchnorm": sync_bn,
        }
        if strategy_arg is not None:
            extra["strategy"] = strategy_arg
        variants.append(dict(extra, use_distributed_sampler=True))
        variants.append(extra)
        slim = dict(extra)
        slim.pop("sync_batchnorm", None)
        variants.append(slim)
    else:
        extra = {
            "precision": precision_arg if isinstance(precision_arg, int) else 32,
            "sync_batchnorm": sync_bn,
        }
        if strategy_arg is not None:
            extra["strategy"] = strategy_arg
        variants.extend(
            [
                {**extra, "accelerator": "gpu" if use_gpu else "cpu", "devices": gpu_devices if use_gpu else 1},
                {**extra, "accelerator": "gpu" if use_gpu else "cpu", "gpus": gpu_devices if use_gpu else 0},
                {**extra, "gpus": gpu_devices if use_gpu else 0},
            ]
        )
        if use_gpu and n_devices > 1:
            variants.append({"gpus": gpu_devices, "accelerator": "ddp", "precision": extra["precision"]})
            variants.append({"gpus": gpu_devices, "distributed_backend": "ddp", "precision": extra["precision"]})
        variants.append({"gpus": gpu_devices if use_gpu else 0, "precision": extra["precision"]})

    last_err: Optional[BaseException] = None
    for extra_kwargs in variants:
        try:
            trainer = pl.Trainer(**trainer_kwargs, **extra_kwargs)
            logger.info(
                "Trainer: accelerator=%s devices=%s strategy=%s precision=%s sync_bn=%s",
                extra_kwargs.get("accelerator", accel),
                extra_kwargs.get("devices", extra_kwargs.get("gpus")),
                extra_kwargs.get("strategy", strategy_arg),
                extra_kwargs.get("precision", precision_arg),
                extra_kwargs.get("sync_batchnorm", False),
            )
            return trainer
        except TypeError as exc:
            last_err = exc
    raise TypeError(f"could not construct pl.Trainer for this Lightning version: {last_err}") from last_err


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_models:
        list_models()
        return 0
    if args.list_folds:
        list_folds()
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    site_label, train_site_names, valid_site_names, eval_ratio = resolve_train_valid_sites(args)

    resume_ckpt: Optional[Path] = None
    if args.ckpt is not None or args.ckpt_dir is not None:
        try:
            resume_ckpt = resolve_checkpoint(args.ckpt, ckpt_dir=args.ckpt_dir)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc

    model_config, model_label = build_model_config(
        model=args.model,
        max_epochs=args.epochs,
        lr=args.lr,
        model_config_path=args.model_config,
        encoder_weights=args.encoder_weights,
        loss_type=args.loss,
        lr_step=args.lr_step,
        recipe=getattr(args, "train_recipe", None),
    )
    model_slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in model_label.lower())

    npz_root = args.npz_root or (args.data_dir / "npz")
    run_name = f"train_{site_label}_{model_slug}"
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_dir or (REPO_ROOT / "output" / run_name)).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("PL compat:", ensure_hardpicks_lightning_compat())
    if not hardpicks_available():
        raise SystemExit("hardpicks (+ torch) required — run setup_lightning.sh / install requirements")

    import hardpicks
    import models.fbp.unet as fbp_unet

    pl.seed_everything(args.seed, workers=True)

    report_root = (args.report_dir or (REPO_ROOT / "report")).resolve()
    report_dir = report_root / f"{run_name}_{run_stamp}"
    rank_zero = env_is_rank_zero()
    report: Optional[TrainingReport] = None
    if rank_zero:
        report = TrainingReport(
            report_dir,
            meta={
                "run_name": run_name,
                "model": model_label,
                "encoder": model_config.get("unet_encoder_type"),
                "fold": site_label if eval_ratio is None else None,
                "train_sites": list(train_site_names),
                "valid_sites": list(valid_site_names),
                "eval_ratio": eval_ratio,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "loss": model_config.get("loss_type"),
                "lr_step": (model_config.get("scheduler_params") or {}).get("step_size"),
                "augmentations": _aug_type_names(
                    getattr(args, "train_augmentations", None) or []
                ),
                "resume_ckpt": str(resume_ckpt) if resume_ckpt else None,
                "backend": args.backend,
                "output_dir": str(output_root),
                "devices": args.devices,
                "strategy": args.strategy,
                "precision": args.precision,
                "train_config": str(getattr(args, "train_config_path", "")),
                "run_stamp": run_stamp,
            },
        )

    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| pl", pl.__version__)
    print("hardpicks", hardpicks.__file__)
    print("FBPUNet module:", fbp_unet.__file__)
    print("Experiment dir:", output_root)
    if report is not None:
        print("Report:", report.report_md)
    if eval_ratio is None:
        print(f"Fold: {site_label} | train: {train_site_names} | valid: {valid_site_names}")
    else:
        print("Sites:", train_site_names, f"| eval_ratio={eval_ratio}")
    print("Model:", model_label, "| encoder:", model_config.get("unet_encoder_type"))
    print("encoder_weights:", model_config.get("encoder_weights"))
    print(
        "loss:",
        model_config.get("loss_type"),
        "| lr_step:",
        (model_config.get("scheduler_params") or {}).get("step_size"),
        "| patience:",
        args.patience,
    )
    augs = getattr(args, "train_augmentations", None) or []
    print("augmentations:", ", ".join(_aug_type_names(augs)) or "none")
    if resume_ckpt is not None:
        print("Resume:", resume_ckpt)
    print("Train config:", getattr(args, "train_config_path", DEFAULT_TRAIN_CONFIG))
    print("DATA_BACKEND:", args.backend, "| NPZ_ROOT:", npz_root)

    config_out = output_root / "model_config.yaml"
    with config_out.open("w") as f:
        yaml.safe_dump(model_config, f, sort_keys=False, default_flow_style=False)
    print("Wrote", config_out)

    recipe_out = output_root / "train_recipe.yaml"
    with recipe_out.open("w") as f:
        yaml.safe_dump(
            {
                "source": str(getattr(args, "train_config_path", "")),
                **dict(getattr(args, "train_recipe", None) or {}),
                "augmentations": getattr(args, "train_augmentations", None) or [],
            },
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    print("Wrote", recipe_out)

    split_out = output_root / "data_split.yaml"
    with split_out.open("w") as f:
        yaml.safe_dump(
            {
                "fold": site_label if eval_ratio is None else None,
                "train_sites": list(train_site_names),
                "valid_sites": list(valid_site_names),
                "eval_ratio": eval_ratio,
            },
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    print("Wrote", split_out)

    tbx_dir = output_root / "tensorboard"
    csv_dir = output_root / "csv_logs"
    tbx_dir.mkdir(exist_ok=True)
    csv_dir.mkdir(exist_ok=True)

    tbx_logger = pl.loggers.TensorBoardLogger(
        save_dir=str(tbx_dir), name="default", default_hp_metric=False
    )
    csv_logger = pl.loggers.CSVLogger(save_dir=str(csv_dir), name="metrics")
    print("TensorBoard:", f"tensorboard --logdir {tbx_dir}")

    try:
        devices_arg = parse_devices(args.devices)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    accel = str(args.accelerator).strip().lower()
    use_gpu = torch.cuda.is_available() and accel != "cpu"
    n_devices = resolved_num_devices(devices_arg, use_gpu=use_gpu)

    train_parser, valid_parser = build_parsers(
        train_site_names=train_site_names,
        valid_site_names=valid_site_names,
        backend=args.backend,
        data_dir=args.data_dir,
        npz_root=npz_root,
        eval_ratio=eval_ratio,
        segm_class_count=SEGMENTATION_CLASS_COUNT,
        augmentations=getattr(args, "train_augmentations", None),
    )
    train_loader, valid_loader = build_loaders(
        train_parser,
        valid_parser,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_gpu,
        drop_last_train=n_devices > 1,
        shuffle_train=n_devices <= 1,
    )
    print(
        f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)} "
        f"| devices={n_devices} | per-GPU batch={args.batch_size} "
        f"| global batch≈{args.batch_size * n_devices}"
    )
    if report is not None:
        report.meta["n_train"] = len(train_parser)
        report.meta["n_valid"] = len(valid_parser)
        report.meta["num_devices"] = n_devices
        report._write_markdown()

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
    progress_cb = ProgressMetricsCallback(
        print_every_n_steps=args.print_every_n_steps,
        report=report,
        checkpoint_cb=checkpoint_cb,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    callbacks: List[Any] = [checkpoint_cb, progress_cb, lr_monitor]
    if args.patience > 0:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=MONITOR_METRIC,
                mode="max",
                patience=int(args.patience),
                verbose=True,
            )
        )

    trainer = make_trainer(
        tbx_logger=tbx_logger,
        csv_logger=csv_logger,
        callbacks=callbacks,
        max_epochs=args.epochs,
        log_every_n_steps=args.log_every_n_steps,
        accelerator=args.accelerator,
        devices=devices_arg,
        strategy=args.strategy,
        precision=args.precision,
        find_unused_parameters=args.find_unused_parameters,
    )

    if use_gpu:
        print(
            f"Device: GPU x{n_devices} "
            f"(visible={torch.cuda.device_count()}, strategy={args.strategy})"
        )
    else:
        print("Device: CPU")
    if rank_zero:
        _batch = next(iter(train_loader))
        print("batch keys:", sorted(_batch.keys()))
        print("samples", tuple(_batch["samples"].shape), _batch["samples"].dtype)

    print(f"Training for up to {args.epochs} epochs (patience={args.patience})…")
    trainer.fit(
        model,
        train_loader,
        valid_loader,
        ckpt_path=str(resume_ckpt) if resume_ckpt is not None else None,
    )

    is_zero = bool(getattr(trainer, "is_global_zero", rank_zero))
    best_path = Path(checkpoint_cb.best_model_path).resolve() if checkpoint_cb.best_model_path else None

    epoch_df = None
    curves_path = None
    if is_zero:
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

    extra_valid: Optional[Dict[str, Any]] = None
    progress_cb.report = None  # don't treat post-fit validate() as another training epoch
    if not args.no_final_validate and best_path and best_path.is_file():
        best_model = _load_fbpunet_from_checkpoint(best_path)
        setattr(best_model, "_tbx_logger", tbx_logger)
        try:
            val_out = trainer.validate(best_model, dataloaders=valid_loader)
        except TypeError:
            val_out = trainer.validate(best_model, val_dataloaders=valid_loader)
        if is_zero:
            print("\nValidation with best checkpoint:")
            if isinstance(val_out, list) and val_out:
                extra_valid = dict(val_out[0])
                for k, v in sorted(extra_valid.items()):
                    print(f"  {k:30s} {v}")
    elif is_zero:
        if args.no_final_validate:
            print("Skipped final validate (--no-final-validate).")
        else:
            print("No best checkpoint on disk; skipped final validate().")

    if is_zero and report is not None:
        report_path = report.finalize(
            best_path=best_path,
            best_score=checkpoint_cb.best_model_score,
            epoch_df=epoch_df,
            extra_valid=extra_valid,
            curves_path=curves_path,
        )
        print("\nDone.")
        print(f"Report: {report_path}")
        print(f"TensorBoard: tensorboard --logdir {tbx_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
