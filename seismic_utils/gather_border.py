"""Border-aware gather transforms that do not depend on a patched hardpicks install.

Lightning / pip installs mila's hardpicks from GitHub. New ops live here so
``linear_time_window`` and extra augs work with that package.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from hardpicks.data.fbp.constants import BAD_FIRST_BREAK_PICK_INDEX
from hardpicks.data.fbp.gather_transforms import _drop_trace_idxs

GatherType = Dict[Any, Any]


def kill_traces(gather: GatherType, prob: float, invalidate_labels: bool = False):
    """Zero traces; optionally mark labels as invalid so CE ignores them."""
    assert 0 < prob < 1
    trace_count = gather["trace_count"]
    kill_mask = np.random.choice([True, False], trace_count, p=[prob, 1 - prob])
    gather["samples"][kill_mask, :] = 0.0
    if invalidate_labels and np.any(kill_mask):
        gather["first_break_labels"][kill_mask] = BAD_FIRST_BREAK_PICK_INDEX
        gather["bad_first_breaks_mask"][kill_mask] = True
        if "first_break_timestamps" in gather:
            gather["first_break_timestamps"][kill_mask] = BAD_FIRST_BREAK_PICK_INDEX


def reverse_polarity(gather: GatherType, prob: float = 0.05):
    """Multiply a random subset of traces by -1 (labels unchanged)."""
    assert 0 <= prob <= 1
    if np.isclose(prob, 0.0):
        return
    trace_count = gather["trace_count"]
    if np.isclose(prob, 1.0):
        gather["samples"] *= -1
        return
    flip_mask = np.random.choice([True, False], trace_count, p=[prob, 1 - prob])
    gather["samples"][flip_mask, :] *= -1


def rebalance_offsets(
    gather: GatherType,
    near_offset_m: Optional[float] = None,
    near_offset_percentile: float = 70.0,
    drop_near_fraction: float = 0.5,
):
    """Drop a fraction of near-offset traces; never drop far-offset traces."""
    assert 0 <= drop_near_fraction <= 1
    assert 0 <= near_offset_percentile <= 100
    if drop_near_fraction <= 0:
        return
    if "offset_distances" not in gather or gather["offset_distances"] is None:
        return
    offsets = np.abs(np.asarray(gather["offset_distances"][:, 0], dtype=np.float64))
    n_traces = int(gather["trace_count"])
    if n_traces <= 1:
        return
    if near_offset_m is None:
        cutoff = float(np.percentile(offsets, near_offset_percentile))
    else:
        cutoff = float(near_offset_m)
    near_idxs = np.where(offsets < cutoff)[0]
    n_drop = int(round(drop_near_fraction * len(near_idxs)))
    n_drop = min(n_drop, len(near_idxs), n_traces - 1)
    if n_drop <= 0:
        return
    to_drop = np.random.choice(near_idxs, n_drop, replace=False)
    _drop_trace_idxs(gather, to_drop.tolist())


def signed_shot_rec_offsets(gather: GatherType) -> np.ndarray:
    """Signed offset along the receiver line (negative = left of the shot)."""
    n_traces = int(gather["trace_count"])
    unsigned = np.abs(np.asarray(gather["offset_distances"][:, 0], dtype=np.float64))
    rec = gather.get("rec_coords")
    shot = gather.get("shot_coords")
    if rec is not None and shot is not None and n_traces >= 2:
        rec_xy = np.asarray(rec, dtype=np.float64).reshape(n_traces, -1)[:, :2]
        shot_xy = np.asarray(shot, dtype=np.float64).reshape(-1)[:2]
        line = rec_xy[-1] - rec_xy[0]
        norm = float(np.linalg.norm(line))
        if norm > 1e-6:
            return (rec_xy - shot_xy) @ (line / norm)
    signed = unsigned.copy()
    shot_idx = int(np.argmin(unsigned)) if n_traces else 0
    signed[:shot_idx] *= -1.0
    return signed


def _labeled_trace_mask(gather: GatherType) -> np.ndarray:
    labels = np.asarray(gather["first_break_labels"])
    valid = labels > BAD_FIRST_BREAK_PICK_INDEX
    if "bad_first_breaks_mask" in gather and gather["bad_first_breaks_mask"] is not None:
        valid = np.logical_and(valid, ~np.asarray(gather["bad_first_breaks_mask"], dtype=bool))
    return valid


def _two_slope_trend_samples(
    signed_offsets: np.ndarray,
    labels: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    n_traces = signed_offsets.shape[0]
    trend = np.zeros(n_traces, dtype=np.float64)
    labeled_idxs = np.where(valid)[0]
    abs_off = np.abs(signed_offsets)
    i0 = labeled_idxs[np.argmin(abs_off[labeled_idxs])]
    x0 = float(signed_offsets[i0])
    t0 = float(labels[i0])
    left_idxs = labeled_idxs[signed_offsets[labeled_idxs] < x0]
    right_idxs = labeled_idxs[signed_offsets[labeled_idxs] > x0]
    p1 = p2 = 0.0
    if len(left_idxs):
        i_l = left_idxs[np.argmin(signed_offsets[left_idxs])]
        dx = float(signed_offsets[i_l] - x0)
        if abs(dx) > 1e-6:
            p1 = (float(labels[i_l]) - t0) / dx
    if len(right_idxs):
        i_r = right_idxs[np.argmax(signed_offsets[right_idxs])]
        dx = float(signed_offsets[i_r] - x0)
        if abs(dx) > 1e-6:
            p2 = (float(labels[i_r]) - t0) / dx
    if len(left_idxs) == 0:
        p1 = p2
    if len(right_idxs) == 0:
        p2 = p1
    left = signed_offsets < x0
    trend[left] = p1 * (signed_offsets[left] - x0) + t0
    trend[~left] = p2 * (signed_offsets[~left] - x0) + t0
    return trend


def _velocity_trend_samples(gather: GatherType, velocity_mps: float) -> np.ndarray:
    dt_ms = float(gather["sample_rate_ms"])
    offsets = np.abs(np.asarray(gather["offset_distances"][:, 0], dtype=np.float64))
    time_ms = 1000.0 * offsets / float(velocity_mps)
    return time_ms / dt_ms


def _warp_samples_to_window(samples: np.ndarray, shift: np.ndarray, window: int) -> np.ndarray:
    n_traces, n_samples = samples.shape
    warped = np.zeros((n_traces, window), dtype=samples.dtype)
    src = np.arange(window, dtype=np.int64)[None, :] + shift.reshape(-1, 1).astype(np.int64)
    valid = (src >= 0) & (src < n_samples)
    trace_ix = np.broadcast_to(np.arange(n_traces)[:, None], src.shape)
    warped[valid] = samples[trace_ix[valid], src[valid]]
    return warped


def apply_linear_time_window(
    gather: GatherType,
    half_window_samples: int = 512,
    min_control_picks: int = 2,
    unlabeled_fallback: str = "skip",
    fallback_velocity_mps: float = 5500.0,
):
    """Crop each gather around a two-slope first-arrival trend.

    Stores ``sample_time_shift`` so original sample index = warped index + shift.
    """
    fallback = str(unlabeled_fallback).strip().lower()
    assert fallback in ("skip", "velocity"), f"invalid unlabeled_fallback: {unlabeled_fallback}"
    assert int(half_window_samples) > 0
    assert int(min_control_picks) >= 1
    n_traces = int(gather["trace_count"])
    if "offset_distances" not in gather or gather["offset_distances"] is None:
        gather["sample_time_shift"] = np.zeros(n_traces, dtype=np.int32)
        return
    samples = np.asarray(gather["samples"])
    labels = np.asarray(gather["first_break_labels"]).copy()
    window = int(2 * int(half_window_samples))
    half = int(half_window_samples)
    valid = _labeled_trace_mask(gather)
    n_control = int(valid.sum())
    if n_control >= int(min_control_picks):
        trend = _two_slope_trend_samples(signed_shot_rec_offsets(gather), labels, valid)
    elif fallback == "velocity":
        trend = _velocity_trend_samples(gather, fallback_velocity_mps)
    else:
        gather["sample_time_shift"] = np.zeros(n_traces, dtype=np.int32)
        return
    shift = np.rint(trend).astype(np.int32) - np.int32(half)
    gather["samples"] = _warp_samples_to_window(samples, shift, window)
    new_labels = labels.astype(np.int64) - shift.astype(np.int64)
    keep = valid & (new_labels >= 0) & (new_labels < window)
    warped_labels = np.full(n_traces, BAD_FIRST_BREAK_PICK_INDEX, dtype=labels.dtype)
    warped_labels[keep] = new_labels[keep].astype(labels.dtype)
    gather["first_break_labels"] = warped_labels
    bad = np.asarray(gather.get("bad_first_breaks_mask", ~valid), dtype=bool).copy()
    bad[~keep] = True
    gather["bad_first_breaks_mask"] = bad
    if "outlier_first_breaks_mask" in gather and gather["outlier_first_breaks_mask"] is not None:
        outlier = np.asarray(gather["outlier_first_breaks_mask"], dtype=bool).copy()
        outlier[~keep] = True
        gather["outlier_first_breaks_mask"] = outlier
    gather["sample_count"] = window
    gather["sample_time_shift"] = shift


def unshift_sample_indices(indices: np.ndarray, sample_time_shift: np.ndarray) -> np.ndarray:
    """Map windowed sample indices back to the original record."""
    return np.asarray(indices, dtype=np.float64) + np.asarray(sample_time_shift, dtype=np.float64)


def generate_windowed_prior_mask(
    gather: GatherType,
    prior_velocity_range,
    prior_offset_range,
):
    """Paint the first-break prior in the current sample axis, honoring ``sample_time_shift``."""
    trace_count, sample_count = gather["trace_count"], gather["sample_count"]
    sample_rate_ms = gather["sample_rate_ms"]
    offset_distances = gather["offset_distances"][:, 0]
    expected_range_sec = np.stack(
        (
            offset_distances / prior_velocity_range[1] + prior_offset_range[0] / 1000,
            offset_distances / prior_velocity_range[0] + prior_offset_range[1] / 1000,
        ),
        axis=1,
    )
    expected_range_idxs = np.round(expected_range_sec / (sample_rate_ms / 1000)).astype(np.int32)
    if "sample_time_shift" in gather and gather["sample_time_shift"] is not None:
        expected_range_idxs = expected_range_idxs - np.asarray(
            gather["sample_time_shift"], dtype=np.int32
        ).reshape(-1, 1)
    fb_mask = np.zeros((trace_count, sample_count), dtype=np.float32)
    for trace_idx, sample_range_idxs in enumerate(expected_range_idxs):
        lo = max(int(sample_range_idxs[0]), 0)
        hi = min(int(sample_range_idxs[1]), sample_count)
        if hi > lo:
            fb_mask[trace_idx, lo:hi] = 1
    gather["first_break_prior"] = fb_mask
