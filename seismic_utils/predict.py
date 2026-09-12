"""Load an FBPUNet checkpoint and predict first-break picks on one gather."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np

from .dataset import ShotGather
from .hardpicks_pl_compat import ensure_hardpicks_lightning_compat

_BAD_FB_LABEL = -1
_DEAD_TRACE_EPS = 1e-8


def _offset_distances(
    n_traces: int,
    offset: np.ndarray | None,
    rec_coords: np.ndarray,
) -> np.ndarray:
    """Hardpicks-style (n_traces, 3) offsets: shot-rec, next-rec, prev-rec."""
    out = np.zeros((n_traces, 3), dtype=np.float32)
    if offset is not None:
        out[:, 0] = np.asarray(offset, dtype=np.float32).reshape(-1)
    if n_traces > 1:
        diffs = np.linalg.norm(np.diff(rec_coords[:, :2], axis=0), axis=1).astype(np.float32)
        out[:-1, 1] = diffs
        out[1:, 2] = diffs
    return out


def resolve_checkpoint(
    path: str | Path | None = None,
    *,
    ckpt_dir: str | Path | None = None,
) -> Path:
    """Resolve an explicit file, the run's best manifest, or one unambiguous file."""
    if path:
        ckpt = Path(path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"checkpoint is not a file: {ckpt}")
        return ckpt
    if ckpt_dir is None:
        raise FileNotFoundError("No checkpoint path or directory provided")
    directory = Path(ckpt_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
    manifest = directory / "best_checkpoint.json"
    if manifest.is_file():
        selected = directory / Path(json.loads(manifest.read_text())["path"]).name
        if not selected.is_file():
            raise FileNotFoundError(f"Best-checkpoint manifest points to missing file: {selected}")
        return selected.resolve()
    matches = sorted(directory.glob("best*.ckpt"))
    if not matches:
        raise FileNotFoundError(f"No best*.ckpt under {directory}")
    if len(matches) != 1:
        raise FileNotFoundError(f"Multiple checkpoints under {directory}; specify --ckpt explicitly")
    return matches[0]


def load_fbp_model(ckpt_path: str | Path, *, device: str | None = None):
    """Load ``FBPUNet`` from a Lightning checkpoint (applies PL2 compat first)."""
    import torch

    ensure_hardpicks_lightning_compat()
    import models.fbp.unet as fbp_unet

    path = Path(ckpt_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is not a file: {path}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = fbp_unet.FBPUNet.load_from_checkpoint(str(path), map_location="cpu")
    model.eval()
    model = model.to(device)
    return model


def decode_fb_picks(raw_preds, model, *, picker: str | None = None, smooth_threshold: int | None = None):
    """Logits → per-trace sample indices."""
    from .pickers import decode_nn_picks

    return decode_nn_picks(raw_preds, model, picker=picker, smooth_threshold=smooth_threshold)


def shot_gather_to_inference_dict(
    gather: ShotGather,
    *,
    origin: str = "viewer",
) -> dict[str, Any]:
    """
    Build a minimal hardpicks-style gather dict from a native ``ShotGather``.

    Avoids opening the hardpicks HDF5 TraceParser (very slow on large sites).
    """
    samples = np.ascontiguousarray(gather.traces, dtype=np.float32)
    n_traces, n_samples = samples.shape
    fb_idx, labeled = gather.first_break_sample_indices()
    fb_labels = fb_idx.copy()
    fb_labels[~labeled] = _BAD_FB_LABEL
    fb_ts = gather.first_breaks_ms.astype(np.float32, copy=True)
    fb_ts[~labeled] = float(_BAD_FB_LABEL)

    rec_coords = np.stack(
        [
            np.asarray(gather.rec_x, dtype=np.float64).reshape(-1),
            np.asarray(gather.rec_y, dtype=np.float64).reshape(-1),
            np.zeros(n_traces, dtype=np.float64),
        ],
        axis=1,
    )
    sample_rate_ms = float(gather.sample_rate_us) / 1000.0
    return {
        "origin": origin,
        "shot_id": int(gather.shot_id),
        "rec_line_id": int(gather.line_id if gather.line_id is not None else -1),
        "rec_ids": np.asarray(gather.channel, dtype=np.int64).reshape(-1),
        "gather_id": int(gather.gather_id if gather.gather_id is not None else -1),
        "gather_trace_ids": np.arange(n_traces, dtype=np.int64),
        "first_break_labels": fb_labels,
        "first_break_timestamps": fb_ts,
        "bad_first_breaks_mask": ~labeled,
        "rec_coords": rec_coords,
        "shot_coords": np.zeros(3, dtype=np.float64),
        "trace_count": n_traces,
        "sample_count": n_samples,
        "sample_rate_ms": sample_rate_ms,
        "dead_rec_mask": np.isclose(samples, 0, atol=_DEAD_TRACE_EPS).all(axis=1),
        "samples": samples,
        "offset_distances": _offset_distances(n_traces, gather.offset, rec_coords),
    }


def _prepare_gather_for_inference(
    gather: dict[str, Any],
    *,
    linear_time_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize samples/offsets like training defaults (no full deepcopy)."""
    from hardpicks.data.fbp.gather_preprocess import ShotLineGatherPreprocessor
    from seismic_utils.gather_border import apply_linear_time_window

    out = dict(gather)
    for key, val in list(out.items()):
        if isinstance(val, np.ndarray):
            out[key] = val.copy()
    if linear_time_window and linear_time_window.get("enabled"):
        params = dict(linear_time_window)
        params.pop("enabled", None)
        apply_linear_time_window(out, **params)
    out["samples"] = ShotLineGatherPreprocessor.normalize_sample_with_tracewise_abs_max_strategy(
        np.asarray(out["samples"], dtype=np.float32)
    )
    if out.get("offset_distances") is not None:
        offsets = np.asarray(out["offset_distances"], dtype=np.float32).copy()
        offsets[:, 0] *= 1.0 / 3000.0
        offsets[:, 1:] *= 1.0 / 50.0
        out["offset_distances"] = offsets
    return out


def _linear_time_window_from_model(model) -> dict[str, Any] | None:
    hp = dict(getattr(model, "hparams", {}) or {})
    td = hp.get("training_data") if isinstance(hp, dict) else None
    site_params = (td or {}).get("site_params") if isinstance(td, dict) else None
    raw = (site_params or {}).get("linear_time_window") if isinstance(site_params, dict) else None
    if not isinstance(raw, dict) or not raw.get("enabled"):
        return None
    return dict(raw)


def predict_first_breaks_ms(
    model,
    hardpicks_gather: dict[str, Any],
    *,
    picker: str | None = None,
    smooth_threshold: int | None = None,
) -> np.ndarray:
    """
    Run the model on one hardpicks-style gather dict.

    Returns predicted first-break times in milliseconds, shape ``(n_traces,)``.
    Invalid / no-pick traces are ``NaN``.

    *picker* selects the decode head (``fbpunet`` vs ``before_after``). Default:
    checkpoint ``hparams.picker`` / ``segm_class_count``.
    """
    import torch
    import hardpicks.data.fbp.data_module as fbp_data_module
    import hardpicks.models.fbp.utils as model_utils

    from seismic_utils.gather_border import unshift_sample_indices

    from .pickers import picker_from_model

    window_cfg = _linear_time_window_from_model(model)
    prepared = _prepare_gather_for_inference(
        hardpicks_gather, linear_time_window=window_cfg
    )
    n_traces = int(prepared["trace_count"])
    dt_ms = float(prepared["sample_rate_ms"])
    resolved_picker = picker or picker_from_model(model)

    batch = fbp_data_module.fbp_batch_collate([prepared], pad_to_nearest_pow2=True)
    with torch.no_grad():
        input_tensor = model_utils.prepare_input_features(
            batch,
            use_dist_offsets=model.use_dist_offsets,
            use_first_break_prior=model.use_first_break_prior,
        ).to(model.device).float()
        logits = model(input_tensor)
        pred_idx, _ = decode_fb_picks(
            logits, model, picker=resolved_picker, smooth_threshold=smooth_threshold
        )

    idx = pred_idx[0, :n_traces].detach().cpu().numpy().astype(np.float64)
    shift = prepared.get("sample_time_shift")
    if shift is not None:
        idx = unshift_sample_indices(idx, np.asarray(shift)[:n_traces])
    fb_ms = idx * dt_ms
    fb_ms[idx <= 0] = np.nan
    return fb_ms


def predict_first_breaks_ms_from_shot_gather(
    model,
    gather: ShotGather,
    *,
    picker: str | None = None,
    smooth_threshold: int | None = None,
) -> np.ndarray:
    """Predict FB times from a native ``ShotGather`` (no hardpicks HDF5 open)."""
    return predict_first_breaks_ms(
        model,
        shot_gather_to_inference_dict(gather),
        picker=picker,
        smooth_threshold=smooth_threshold,
    )
