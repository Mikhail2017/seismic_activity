"""Aggregate before / after / unlabeled sample counts across HDF5 assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import (
    SAMP_RATE_KEY,
    TRACE_GROUP,
    _as_1d,
    _normalize_first_breaks,
    list_dataset_files,
    open_trace_group,
)
from .export_npz import infer_asset_name
from .sites import resolve_site_config


@dataclass(frozen=True)
class ClassCounts:
    """Sample counts matching the Gradio before / after / unlabeled regions."""

    before: int
    after: int
    unlabeled: int
    n_traces: int
    n_samples: int
    asset: str
    path: str

    @property
    def total(self) -> int:
        return self.before + self.after + self.unlabeled


def count_classes_in_hdf5(path: str | Path, *, asset: str | None = None) -> ClassCounts:
    """
    Count samples before / after the first break and on unlabeled traces.

    Uses the same rule as the viewer overlay:
    - labeled & time < fb  → before
    - labeled & time >= fb → after
    - unlabeled traces     → all samples unlabeled
    """
    path = Path(path)
    site = resolve_site_config(path)
    asset_name = asset or site.site_name
    fb_key = site.first_break_field_name
    handle, group = open_trace_group(path)
    try:
        if fb_key not in group:
            raise KeyError(f"Missing '{fb_key}' in {TRACE_GROUP}")

        fb_ms = _normalize_first_breaks(_as_1d(group[fb_key][()]))
        n_traces = int(fb_ms.shape[0])

        if "data_array" in group:
            n_samples = int(group["data_array"].shape[1])
        elif "SAMP_NUM" in group:
            n_samples = int(_as_1d(group["SAMP_NUM"][:1])[0])
        else:
            raise KeyError("Cannot determine sample count (need data_array or SAMP_NUM)")

        if SAMP_RATE_KEY in group:
            sample_rate_us = float(_as_1d(group[SAMP_RATE_KEY][:1])[0])
        else:
            sample_rate_us = 1000.0

        time_ms = np.arange(n_samples, dtype=np.float64) * (sample_rate_us / 1000.0)
        labeled = np.isfinite(fb_ms)
        # searchsorted(..., side="left") == number of times with time < fb
        before_per_trace = np.zeros(n_traces, dtype=np.int64)
        before_per_trace[labeled] = np.searchsorted(time_ms, fb_ms[labeled], side="left")
        before_per_trace = np.clip(before_per_trace, 0, n_samples)

        after_per_trace = np.zeros(n_traces, dtype=np.int64)
        after_per_trace[labeled] = n_samples - before_per_trace[labeled]

        unlabeled_per_trace = np.where(labeled, 0, n_samples).astype(np.int64)

        return ClassCounts(
            before=int(before_per_trace.sum()),
            after=int(after_per_trace.sum()),
            unlabeled=int(unlabeled_per_trace.sum()),
            n_traces=n_traces,
            n_samples=n_samples,
            asset=asset_name,
            path=str(path),
        )
    finally:
        handle.close()


def count_classes_in_directory(data_dir: str | Path) -> list[ClassCounts]:
    """Count classes for every HDF5 asset under *data_dir*."""
    files = list_dataset_files(data_dir)
    return [count_classes_in_hdf5(path) for path in files]
