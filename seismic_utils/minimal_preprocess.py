"""Amplitude preprocess, windowed FB masks, and ceil-16 collate for the Meneses recipe."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.utils.data

from .minimal_split import BAD_PICK_INDEX, GatherKey, gather_key

DONTCARE = -1
PAD_MULTIPLE = 16
PAD_VALUE = 1.0
ZSCORE_EPS = 1e-6
QUANTIZE_CLIP_SIGMA = 8.0
LABEL_WINDOW_MS = 10.0


def uses_minimal_preprocess(hp: Mapping[str, Any] | None) -> bool:
    return bool((hp or {}).get("minimal_annotations"))


def window_half_samples(sample_rate_ms: float, window_ms: float = LABEL_WINDOW_MS) -> int:
    if float(window_ms) <= 0:
        return 0
    dt = float(sample_rate_ms)
    if not np.isfinite(dt) or dt <= 0:
        dt = 2.0
    return max(1, int(round(float(window_ms) / dt)))


def tracewise_zscore(samples: np.ndarray, *, eps: float = ZSCORE_EPS) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"samples must be (traces, time), got {arr.shape}")
    mean = np.nanmean(arr, axis=1, keepdims=True)
    std = np.nanstd(arr, axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    out = (arr - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def quantize_int16(samples: np.ndarray, *, clip_sigma: float = QUANTIZE_CLIP_SIGMA) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    clipped = np.clip(arr, -float(clip_sigma), float(clip_sigma))
    scale = float(clip_sigma) if clip_sigma else 1.0
    q = np.rint(clipped / scale * 32767.0)
    q = np.clip(q, -32768, 32767)
    return (q / 32767.0 * scale).astype(np.float32)


def apply_amplitude_preprocess(samples: np.ndarray) -> np.ndarray:
    return quantize_int16(tracewise_zscore(samples))


def windowed_fb_mask(
    first_break_labels: np.ndarray,
    n_samples: int,
    half_width: int,
) -> np.ndarray:
    labels = np.asarray(first_break_labels).reshape(-1)
    n_traces = int(labels.shape[0])
    mask = np.full((n_traces, int(n_samples)), DONTCARE, dtype=np.int32)
    hw = max(int(half_width), 0)
    for i, fb in enumerate(labels):
        fb_i = int(fb)
        if fb_i <= BAD_PICK_INDEX:
            continue
        lo = max(fb_i - hw, 0)
        hi = min(fb_i + 1 + hw, int(n_samples))
        if hi <= lo:
            continue
        mask[i, :] = 0
        mask[i, lo:hi] = 1
    return mask


def strip_ground_truth(item: dict[str, Any]) -> dict[str, Any]:
    out = dict(item)
    labels = np.asarray(out.get("first_break_labels", []), dtype=np.int32).reshape(-1)
    n_traces = int(labels.shape[0] or int(out.get("trace_count") or 0))
    n_samples = int(out.get("sample_count") or np.asarray(out.get("samples", np.zeros((n_traces, 1)))).shape[-1])
    out["first_break_labels"] = np.zeros(n_traces, dtype=np.int32)
    ts = np.asarray(out.get("first_break_timestamps", np.zeros(n_traces)), dtype=np.float32).reshape(-1)
    out["first_break_timestamps"] = np.zeros_like(ts)
    out["bad_first_breaks_mask"] = np.ones(n_traces, dtype=bool)
    out["segmentation_mask"] = np.full((n_traces, n_samples), DONTCARE, dtype=np.int32)
    return out


def apply_pseudo_labels(
    item: dict[str, Any],
    picks: np.ndarray,
    *,
    half_width: int,
) -> dict[str, Any]:
    out = dict(item)
    samples = np.asarray(out["samples"])
    n_traces, n_samples = int(samples.shape[0]), int(samples.shape[1])
    pred = np.asarray(picks, dtype=np.float64).reshape(-1)[:n_traces]
    if pred.shape[0] < n_traces:
        pred = np.pad(pred, (0, n_traces - pred.shape[0]), constant_values=np.nan)
    labels = np.zeros(n_traces, dtype=np.int32)
    good = np.isfinite(pred) & (pred > BAD_PICK_INDEX)
    labels[good] = np.rint(pred[good]).astype(np.int32)
    labels = np.clip(labels, 0, n_samples - 1)
    labels[~good] = BAD_PICK_INDEX
    out["first_break_labels"] = labels
    out["bad_first_breaks_mask"] = labels <= BAD_PICK_INDEX
    out["segmentation_mask"] = windowed_fb_mask(labels, n_samples, half_width)
    return out


def _copy_item(item: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in item.items():
        if isinstance(value, np.ndarray):
            out[key] = np.array(value, copy=True)
        else:
            out[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value
    return out


class MinimalAnnotationDataset(torch.utils.data.Dataset):
    """Index subset that applies Meneses amplitude/label rules on top of a parser."""

    def __init__(
        self,
        base,
        indices: Sequence[int],
        *,
        mode: str,
        window_ms: float = LABEL_WINDOW_MS,
        apply_amp: bool = True,
        pseudo_picks: Optional[Mapping[GatherKey, np.ndarray]] = None,
    ):
        if mode not in {"labeled", "unlabeled", "oracle", "pseudo"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.base = base
        self.indices = [int(i) for i in indices]
        self.mode = mode
        self.window_ms = float(window_ms)
        self.apply_amp = bool(apply_amp)
        self.pseudo_picks = dict(pseudo_picks or {})

    def __len__(self) -> int:
        return len(self.indices)

    def get_meta_gather(self, gather_id: int) -> dict[str, Any]:
        return self.base.get_meta_gather(self.indices[gather_id])

    def __getitem__(self, gather_id: int) -> dict[str, Any]:
        item = _copy_item(self.base[self.indices[gather_id]])
        samples = np.asarray(item["samples"], dtype=np.float32)
        if self.apply_amp:
            item["samples"] = apply_amplitude_preprocess(samples)
        else:
            item["samples"] = samples
        n_samples = int(item["samples"].shape[1])
        item["sample_count"] = n_samples
        dt = float(item.get("sample_rate_ms") or 2.0)
        half = window_half_samples(dt, self.window_ms)
        if self.mode == "unlabeled":
            item = strip_ground_truth(item)
        elif self.mode == "pseudo":
            key = gather_key(item)
            picks = self.pseudo_picks.get(key)
            if picks is None:
                item = strip_ground_truth(item)
            else:
                item = apply_pseudo_labels(item, picks, half_width=half)
        elif self.mode == "labeled":
            item["segmentation_mask"] = windowed_fb_mask(
                item["first_break_labels"], n_samples, half
            )
        else:  # oracle: keep GT, point labels (eval metrics use pick times)
            item["segmentation_mask"] = windowed_fb_mask(
                item["first_break_labels"], n_samples, 0
            )
        return item


def ceil_to_multiple(n: int, multiple: int = PAD_MULTIPLE) -> int:
    m = max(int(multiple), 1)
    n = max(int(n), 1)
    return int(math.ceil(n / m) * m)


def _max_dim(batch, fields, dim: int) -> int:
    max_n = 0
    for sample in batch:
        for field in fields:
            arr = sample.get(field)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim > dim:
                max_n = max(max_n, int(arr.shape[dim]))
    return max_n


def minimal_batch_collate(batch, *, pad_multiple: int = PAD_MULTIPLE, samples_pad_value: float = PAD_VALUE):
    """Pad a batch to ceil(max, 16); sample amplitudes pad with ``samples_pad_value``."""
    from hardpicks.data.fbp.collate import get_fields_to_double_pad, get_fields_to_pad

    fields_to_pad = list(get_fields_to_pad())
    double_pad = set(get_fields_to_double_pad())
    max_traces = ceil_to_multiple(_max_dim(batch, [f for f, _ in fields_to_pad], 0), pad_multiple)
    max_samples = ceil_to_multiple(_max_dim(batch, list(double_pad), 1), pad_multiple)
    padded = []
    for sample in batch:
        item = dict(sample)
        for field, pad_value in fields_to_pad:
            if field not in item or item[field] is None:
                continue
            arr = np.asarray(item[field])
            fill = samples_pad_value if field == "samples" else pad_value
            if field in double_pad:
                pad_w = (
                    (0, max_traces - arr.shape[0]),
                    (0, max_samples - arr.shape[1]),
                )
                item[field] = np.pad(arr, pad_w, mode="constant", constant_values=fill)
            else:
                extra = [(0, 0) for _ in range(1, arr.ndim)]
                item[field] = np.pad(
                    arr,
                    [(0, max_traces - arr.shape[0]), *extra],
                    mode="constant",
                    constant_values=fill,
                )
        padded.append(item)
    output = torch.utils.data._utils.collate.default_collate(padded)
    output["batch_size"] = len(batch)
    return output
