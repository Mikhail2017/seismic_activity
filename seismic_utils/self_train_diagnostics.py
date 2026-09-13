"""Observation-only artifacts for minimal-annotation self-training.

These artifacts describe the current implementation, not the intended paper
recipe. They are for comparison/audit, not an interrupted-training resume API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from .fbp_eval_report import paper_pick_metrics
from .minimal_preprocess import (
    DONTCARE, PAD_MULTIPLE, PAD_VALUE, QUANTIZE_CLIP_SIGMA, ZSCORE_EPS,
    ceil_to_multiple, window_half_samples,
)
from .minimal_split import dict_to_key, key_to_dict, split_keys
from .pseudo_label_qc import MIN_SURVIVE_FRAC, N_OFFSET_BINS, N_SIGMA

SCHEMA_VERSION = 1
DECODER_VERSION = "argmax-fb-unpicked-unmasked-sigmoid-fb-v1"


def json_value(value):
    """Strict JSON, with missing/nonfinite numeric values represented by null."""
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, str):
        return str(value)  # e.g. torch.torch_version.TorchVersion for YAML's safe dumper
    if torch.is_tensor(value):
        return json_value(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def content_digest(value) -> str:
    payload = json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json(path: Path, value) -> Path:
    """Atomically replace one artifact; readers never see partial JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(json_value(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return path.resolve()


def code_identity(root: Path) -> dict[str, Any]:
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, timeout=10,
            ).decode().strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    # Include working source hashes: HEAD alone does not identify an edited run.
    paths = [
        *root.glob("train/*.py"), *root.glob("seismic_utils/*.py"),
        *root.glob("models/**/*.py"),
        *root.glob("src/hardpicks/hardpicks/**/*.py"),
    ]
    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "status": status,
        "source_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
        },
    }


def describe_geometry(parser, indices, window_ms: float) -> dict[str, Any]:
    """Read geometry only; no pool annotations or amplitude reads for diagnostics."""
    shapes, windows = Counter(), Counter()
    for index in indices:
        meta = parser.get_meta_gather(index)
        shape = (int(meta["trace_count"]), int(meta["sample_count"]))
        shapes[shape] += 1
        dt = float(meta.get("sample_rate_ms") or 2.0)
        windows[(dt, window_half_samples(dt, window_ms))] += 1
    return {
        "n_gathers": len(indices),
        "real_shapes": [
            {"traces": t, "samples": s, "n_gathers": n} for (t, s), n in sorted(shapes.items())
        ],
        "batch_padding_upper_bound": [
            ceil_to_multiple(max(shape[axis] for shape in shapes)) for axis in (0, 1)
        ] if shapes else None,
        "training_windows": [
            {"sample_rate_ms": dt, "half_width_samples": half, "n_gathers": n}
            for (dt, half), n in sorted(windows.items())
        ],
    }


def effective_protocol(model_config, window_ms, monitor) -> dict[str, Any]:
    """Resolved semantics, including constants the entry point ignores in YAML."""
    loss_params = model_config.get("loss_params") or {}
    return {
        "window": {"units": "ms", "half_width": window_ms,
                   "validation_loss_half_width_samples": 0, "timing_targets": "manual_points"},
        "preprocessing": {"order": ["trace_zscore", "int16_quantize", "pad"],
                          "zscore_eps": ZSCORE_EPS, "quantize_clip_sigma": QUANTIZE_CLIP_SIGMA},
        "padding": {"policy": "batch_max_ceil_multiple", "fixed_shape": None,
                    "axes": ["traces", "samples"], "multiple": PAD_MULTIPLE,
                    "amplitude_value": PAD_VALUE, "target_value": DONTCARE},
        "loss": {"type": model_config["loss_type"], "params": loss_params,
                 "reduction": loss_params.get("reduction", "mean"),
                 "normalization": "sum_target_class_weights" if model_config["loss_type"] == "crossentropy"
                                  and loss_params.get("reduction", "mean") == "mean" else "loss_specific",
                 "ignore_index": DONTCARE, "missing_manual_labels": "ignore_trace",
                 "rejected_pseudo_picks": "ignore_trace", "pseudo_out_of_range": "clip_to_last_sample"},
        "decoder": {"version": DECODER_VERSION, "pick": "argmax_raw_fb_logit",
                    "unpicked": "background_wins_everywhere_or_index_zero",
                    "masks_padding": False, "confidence": "sigmoid_fb_logit"},
        "checkpoint": {"monitor": monitor, "mode": "max", "save_top_k": 1,
                       "continuation": "best_weights_only", "teacher": "best_selected_checkpoint",
                       "optimizer_restart": "every_inner_fit", "scheduler_restart": "every_inner_fit",
                       "missing_checkpoint": "reuse_previous_if_available",
                       "reset_condition": "after_expansion_only"},
        "qc": {"n_bins": N_OFFSET_BINS, "n_sigma": N_SIGMA,
               "min_survive_frac": MIN_SURVIVE_FRAC, "denominator": "all_real_traces"},
        "evaluation": {"partition": "unlabeled_pool", "targets": "original_manual_picks",
                       "includes_admitted_pseudo_gathers": True, "batch_size": 1},
    }


def split_identity(split) -> dict[str, Any]:
    return {
        "sha256": content_digest(split), "seed": split.get("seed"), "site": split.get("site"),
        "counts": {name: len(split_keys(split, name)) for name in
                   ("labeled_train", "labeled_val", "unlabeled_pool")},
    }


class AnnotationSourceDataset(torch.utils.data.Dataset):
    """Add only provenance to a training view; never consult raw parser labels."""

    def __init__(self, dataset, source: int):
        self.dataset, self.source = dataset, int(source)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "_annotation_source": self.source}

    def get_meta_gather(self, index):
        return self.dataset.get_meta_gather(index)


def batch_pixel_counts(batch) -> dict[str, dict[str, int]]:
    """Count actual padded training targets, split by annotation source."""
    targets = batch["segmentation_mask"]
    result = {}
    for source_id, name in ((0, "manual"), (1, "pseudo")):
        selected = batch["_annotation_source"] == source_id
        masks = targets[selected]
        positive = int((masks == 1).sum().item())
        background = int((masks == 0).sum().item())
        real = int((batch["trace_count"][selected].long() * batch["sample_count"][selected].long()).sum().item())
        result[name] = {
            "gather_exposures": int(selected.sum().item()),
            "positive": positive, "background": background,
            "ignored_real": real - positive - background,
            "ignored_padding": masks.numel() - real,
            "ignored": masks.numel() - positive - background,
        }
    return result


def merge_counts(counts):
    result = {}
    for block in counts:
        for source, values in block.items():
            dest = result.setdefault(source, Counter())
            dest.update(values)
    return {source: dict(values) for source, values in result.items()}


def _all_ranks(value):
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()):
        return [value]
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, value)
    return parts


def optimizer_diagnostics(optimizers) -> list[dict[str, Any]]:
    result = []
    for optimizer in optimizers:
        steps = [int(state["step"]) for state in optimizer.state.values() if "step" in state]
        result.append({
            "type": type(optimizer).__name__,
            "step_min": min(steps) if steps else None, "step_max": max(steps) if steps else None,
            "parameter_groups": [{k: v for k, v in group.items() if k != "params"}
                                 for group in optimizer.param_groups],
        })
    return json_value(result)


class SelfTrainDiagnostics(pl.Callback):
    """Observe existing train/validation batches without extra model forwards."""

    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.epochs: dict[int, dict] = {}
        self.counts: dict = {}
        self.shapes = Counter()
        self.runtime = {}

    def on_fit_start(self, trainer, pl_module):
        loss = pl_module.loss_fn
        self.runtime = json_value({
            "device": str(pl_module.device), "world_size": trainer.world_size,
            "precision": trainer.precision, "strategy": type(trainer.strategy).__name__,
            "loss": {"class": type(loss).__name__, "reduction": getattr(loss, "reduction", None),
                     "ignore_index": getattr(loss, "ignore_index", None),
                     "weight": getattr(loss, "weight", None)},
        })

    def on_train_epoch_start(self, trainer, pl_module):
        self.counts, self.shapes = {}, Counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.counts = merge_counts([self.counts, batch_pixel_counts(batch)])
        self.shapes["x".join(str(n) for n in batch["samples"].shape[-2:])] += 1

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        evaluator = pl_module.valid_evaluator
        evaluator.finalize()
        frame = evaluator._dataframe[["Errors", "Predictions"]]
        frame = pd.concat(_all_ranks(frame), ignore_index=True)
        metrics = paper_pick_metrics(frame)
        pred = frame["Errors"].notna() & (frame["Predictions"] > 0)
        errors = frame.loc[pred, "Errors"]
        metrics.update({
            "Bias": float(errors.mean()) if len(errors) else None,
            "P95AbsError": float(errors.abs().quantile(0.95)) if len(errors) else None,
            "P99AbsError": float(errors.abs().quantile(0.99)) if len(errors) else None,
        })
        row = self.epochs.setdefault(int(trainer.current_epoch), {})
        row["validation"] = json_value(metrics)

    def on_train_epoch_end(self, trainer, pl_module):
        row = self.epochs.setdefault(int(trainer.current_epoch), {})
        row.update({
            "epoch": int(trainer.current_epoch), "global_step": int(trainer.global_step),
            "pixels": merge_counts(_all_ranks(self.counts)),
            "batch_shapes": dict(sum((Counter(s) for s in _all_ranks(dict(self.shapes))), Counter())),
            "optimizers": optimizer_diagnostics(trainer.optimizers),
        })
        if trainer.is_global_zero:
            write_json(self.output_dir / "epoch_diagnostics.json", {
                "schema_version": SCHEMA_VERSION, "runtime": self.runtime,
                "pixel_count_scope": "actual_training_exposures_per_epoch_including_padding",
                "epochs": list(self.epochs.values()),
            })

    def fit_summary(self, trainer, best, checkpoint):
        selected_epoch = checkpoint.get("epoch") if checkpoint else None
        return {
            "schema_version": SCHEMA_VERSION,
            "pixel_count_scope": "actual_training_exposures_all_epochs_including_padding",
            "runtime": self.runtime,
            "terminal_global_step": int(trainer.global_step),
            "terminal_optimizers": optimizer_diagnostics(trainer.optimizers),
            "selected_checkpoint": str(best) if best else None,
            "selected_epoch": selected_epoch,
            "selected_global_step": checkpoint.get("global_step") if checkpoint else None,
            "validation": self.epochs.get(selected_epoch, {}).get("validation"),
            "terminal_validation": list(self.epochs.values())[-1].get("validation") if self.epochs else None,
            "pixels": merge_counts(row.get("pixels", {}) for row in self.epochs.values()),
            "epochs": list(self.epochs.values()),
        }


def prediction_counts(picks, sample_count: int) -> dict[str, int]:
    pred = np.asarray(picks, dtype=np.float64)
    finite = np.isfinite(pred)
    nonpositive = finite & (pred <= 0)
    out_of_range = finite & (pred >= sample_count)
    return {
        "n_traces": int(pred.size), "n_predicted": int((finite & (pred > 0)).sum()),
        "n_nonfinite": int((~finite).sum()), "n_unpicked_or_nonpositive": int(nonpositive.sum()),
        "n_out_of_range": int(out_of_range.sum()),
        "n_invalid": int(((~finite) | nonpositive | out_of_range).sum()),
    }


def write_pseudo_shard(path, accepted, qc_stats, *, iteration, teacher, split_sha256):
    """Persist raw QC outputs (including existing out-of-range behavior), not GT."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"Pseudo-label artifact already exists: {path}")
    by_key = {tuple(row["key"]): row for row in qc_stats}
    rows = [{"key": key_to_dict(key), "picks_samples": picks,
             "qc": by_key[key]} for key, picks in accepted.items()]
    payload = {
        "schema_version": SCHEMA_VERSION, "iteration": iteration,
        "teacher_checkpoint": str(Path(teacher).resolve()), "decoder_version": DECODER_VERSION,
        "split_sha256": split_sha256, "pick_order": "qc.receiver_ids",
        "missing_pick": None, "gathers": rows,
    }
    write_json(path, payload)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "iteration": iteration, "n_gathers": len(rows),
            "n_picks": sum(int(np.isfinite(p).sum()) for p in accepted.values())}


def load_pseudo_archive(index_path):
    """Read audit picks and provenance, verifying immutable shard identities."""
    index = json.loads(Path(index_path).read_text())
    picks, provenance = {}, {}
    for shard in index["shards"]:
        raw = Path(shard["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != shard["sha256"]:
            raise ValueError(f"Pseudo-label checksum mismatch: {shard['path']}")
        data = json.loads(raw)
        if data["split_sha256"] != index["split_sha256"]:
            raise ValueError("Pseudo-label split identity mismatch")
        for row in data["gathers"]:
            key = dict_to_key(row["key"])
            if key in picks:
                raise ValueError(f"Duplicate pseudo-label key: {key}")
            picks[key] = np.asarray(row["picks_samples"], dtype=np.float64)
            provenance[key] = {"iteration": data["iteration"],
                               "teacher_checkpoint": data["teacher_checkpoint"], "qc": row["qc"]}
    return picks, provenance