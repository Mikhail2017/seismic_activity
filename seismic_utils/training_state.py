"""Checkpoint compatibility checks and run artifact identity."""

import ast
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from seismic_utils.pickers import picker_from_hparams


PREPROCESSING_VERSION = "owned-hdf5-metadata-v1"


def load_checkpoint(path):
    # Training checkpoints are trusted local artifacts, not arbitrary uploads.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def validate_resume_checkpoint(checkpoint, config):
    """Reject semantic changes that strict tensor loading cannot detect."""
    saved = checkpoint.get("hyper_parameters", {})
    if picker_from_hparams(saved) != picker_from_hparams(config):
        raise ValueError("Resume picker differs from checkpoint; use matching settings or --init-ckpt")
    keys = (
        "segm_class_count", "unet_encoder_type", "unet_decoder_type", "encoder_block_count",
        "encoder_block_channels", "decoder_block_channels", "mid_block_channels",
        "decoder_attention_type", "use_dist_offsets", "use_first_break_prior", "coordconv",
        "loss_type", "loss_params", "optimizer_type", "optimizer_params", "scheduler_type",
        "scheduler_params", "update_scheduler_at_epochs", "segm_first_break_prob_threshold",
        "segm_first_break_smooth_threshold", "training_data",
    )
    def normalized(value):
        if isinstance(value, str) and value.startswith("["):
            return ast.literal_eval(value)
        return value
    mismatches = [key for key in keys if normalized(saved.get(key)) != normalized(config.get(key))]
    if mismatches:
        raise ValueError("Incompatible resume settings: " + ", ".join(mismatches))
    if "seismic_scheduler_state" not in checkpoint:
        raise ValueError("Checkpoint lacks scheduler state; use --init-ckpt for weights-only initialization")
    if saved.get("scheduler_type") == "LinearWarmupCosineAnnealingLR" and saved.get("max_epochs") != config.get("max_epochs"):
        raise ValueError("Cannot change max_epochs when resuming a fixed-horizon warmup/cosine schedule")


def run_token(default):
    """A shared ID for torchrun workers; inherited by Lightning subprocesses."""
    token = os.environ.get("SEISMIC_RUN_ID")
    if not token:
        elastic_id = os.environ.get("TORCHELASTIC_RUN_ID")
        if elastic_id and elastic_id != "none":
            token = hashlib.sha256(elastic_id.encode()).hexdigest()[:16]
        elif int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("Set SEISMIC_RUN_ID to a unique shared ID for this distributed run")
        else:
            token = default
    if not token or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in token):
        raise ValueError("SEISMIC_RUN_ID must contain only letters, digits, underscores, or hyphens")
    os.environ["SEISMIC_RUN_ID"] = token
    return token


def reserve_run_directory(path):
    """Refuse to overwrite old checkpoints, configs, or logs."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new --output-dir.")
    with (path / "run.json").open("x") as f:
        json.dump({"run_id": os.environ.get("SEISMIC_RUN_ID"), "pid": os.getpid()}, f)


def wait_for_run_directory(path, timeout=90):
    """Do not let nonzero workers race rank zero's empty-directory check."""
    marker = Path(path) / "run.json"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if json.loads(marker.read_text()).get("run_id") == os.environ.get("SEISMIC_RUN_ID"):
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.1)
    raise RuntimeError(f"Rank zero did not initialize the run directory: {path}")