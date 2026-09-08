"""Optional adapters that reuse hardpicks' official shot×line gather parser.

The original task asks to separate each 2D seismic image (a sequence of traces)
using receiver geometry. hardpicks implements that as **shot ∩ receiver-line**
gathers via ``ShotLineGatherDataset``. This module wraps that API into the
``ShotGather`` / ``LineGatherRef`` types used by our viewer and NPZ export.

Falls back cleanly when hardpicks is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DEFAULT_DATA_DIR, LineGatherRef, ShotGather, ensure_hdf5_path
from .sites import SiteConfig, resolve_site_config


def hardpicks_available() -> bool:
    """Return True if hardpicks gather parsing can be imported."""
    try:
        import hardpicks.data.fbp.gather_parser  # noqa: F401
        return True
    except Exception:
        return False


def resolve_hardpicks_site_info(
    site_name: str,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Build a hardpicks ``site_info`` dict pointed at this repo's flat data layout.

    hardpicks expects paths like ``<data_dir>/Lalor_3D/...hdf5``; our files live
    directly under ``DEFAULT_DATA_DIR``. This remaps ``raw_hdf5_path`` /
    ``processed_hdf5_path`` accordingly while keeping digit counts and FB fields.
    """
    from hardpicks.data.fbp import site_info as fbp_site_info

    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    site = resolve_site_config(site_name=site_name)
    info = dict(fbp_site_info.get_site_info_by_name(site_name, data_dir=str(root)))
    hdf5 = (root / site.hdf5_file_name).resolve()
    if not hdf5.is_file():
        raise FileNotFoundError(f"HDF5 for {site_name} not found at {hdf5}")
    info["raw_hdf5_path"] = str(hdf5)
    info["processed_hdf5_path"] = str(hdf5)
    info["first_break_field_name"] = site.first_break_field_name
    info["receiver_id_digit_count"] = site.receiver_id_digit_count
    return info


def open_hardpicks_dataset(
    hdf5_path: str | Path,
    *,
    site: SiteConfig | None = None,
    site_name: str | None = None,
    convert_to_fp16: bool = False,
    convert_to_int16: bool = False,
    provide_offset_dists: bool = True,
    cache_trace_metadata: bool = True,
):
    """
    Open an official hardpicks ``ShotLineGatherDataset``.

    Defaults disable fp16/int16 casting so exploration/export keep full precision.
    """
    from hardpicks.data.fbp.gather_parser import create_shot_line_gather_dataset

    path = ensure_hdf5_path(hdf5_path)
    site_cfg = site or resolve_site_config(path, site_name=site_name)
    return create_shot_line_gather_dataset(
        hdf5_path=str(path),
        site_name=site_cfg.site_name,
        receiver_id_digit_count=site_cfg.receiver_id_digit_count,
        first_break_field_name=site_cfg.first_break_field_name,
        convert_to_fp16=convert_to_fp16,
        convert_to_int16=convert_to_int16,
        preload_trace_data=False,
        cache_trace_metadata=cache_trace_metadata,
        provide_offset_dists=provide_offset_dists,
    )


def hardpicks_item_to_shot_gather(item: dict[str, Any]) -> ShotGather:
    """
    Convert a hardpicks gather dict into our ``ShotGather``.

    Unlabeled convention: hardpicks uses timestamps ``<= 0`` / labels ``-1``;
    we map those to ``NaN`` in ``first_breaks_ms`` for plotting/export.
    """
    samples = np.asarray(item["samples"], dtype=np.float32)
    if samples.ndim != 2:
        raise ValueError(f"Expected samples (n_traces, n_samples), got {samples.shape}")

    fb_ms = np.asarray(item["first_break_timestamps"], dtype=np.float64).reshape(-1)
    bad = np.asarray(item.get("bad_first_breaks_mask"), dtype=bool).reshape(-1)
    if bad.shape != fb_ms.shape:
        bad = fb_ms <= 0
    fb_ms = fb_ms.copy()
    fb_ms[bad] = np.nan
    fb_ms[fb_ms <= 0] = np.nan

    n_traces = samples.shape[0]
    rec_coords = item.get("rec_coords")
    if rec_coords is not None:
        coords = np.asarray(rec_coords, dtype=np.float64).reshape(n_traces, -1)
        rec_x = coords[:, 0]
        rec_y = coords[:, 1]
    else:
        rec_x = np.zeros(n_traces, dtype=np.float64)
        rec_y = np.zeros(n_traces, dtype=np.float64)

    offset = None
    offset_distances = item.get("offset_distances")
    if offset_distances is not None:
        # hardpicks stacks [shot-rec, next-rec, prev-rec] offsets
        offset = np.asarray(offset_distances, dtype=np.float64).reshape(n_traces, -1)[:, 0]

    sample_rate_ms = float(item["sample_rate_ms"])
    sample_rate_us = sample_rate_ms * 1000.0

    # Keep hardpicks gather order (intersect1d / file order within the line).
    channel = np.arange(n_traces, dtype=np.int64)

    return ShotGather(
        shot_id=int(item["shot_id"]),
        traces=samples,
        first_breaks_ms=fb_ms,
        channel=channel,
        rec_x=rec_x,
        rec_y=rec_y,
        sample_rate_us=sample_rate_us,
        offset=offset,
        line_id=int(item["rec_line_id"]),
        gather_id=int(item["gather_id"]),
    )


def build_line_gather_index_hardpicks(
    hdf5_path: str | Path,
    *,
    site: SiteConfig | None = None,
    dataset=None,
) -> tuple[list[LineGatherRef], Any]:
    """
    Build ``LineGatherRef`` list from a hardpicks dataset (exact official split).

    Returns ``(refs, dataset)`` so callers can keep the open parser for loading.
    """
    ds = dataset or open_hardpicks_dataset(hdf5_path, site=site)
    refs: list[LineGatherRef] = []
    for gather_id in range(len(ds)):
        meta = ds.get_meta_gather(gather_id)
        refs.append(
            LineGatherRef(
                gather_id=int(gather_id),
                shot_id=int(meta["shot_id"]),
                line_id=int(meta["rec_line_id"]),
                trace_indices=np.asarray(meta["gather_trace_ids"], dtype=np.int64),
            )
        )
    return refs, ds


def load_line_gather_hardpicks(
    dataset,
    gather_id: int,
) -> ShotGather:
    """Load one gather from an open hardpicks dataset."""
    return hardpicks_item_to_shot_gather(dataset[gather_id])


class HardpicksGatherStore:
    """Cached hardpicks-backed gather access for viewer/export."""

    def __init__(self, hdf5_path: str | Path, *, site: SiteConfig | None = None):
        self.path = Path(ensure_hdf5_path(hdf5_path))
        self.site = site or resolve_site_config(self.path)
        self.dataset = open_hardpicks_dataset(self.path, site=self.site)
        self.refs, _ = build_line_gather_index_hardpicks(
            self.path, site=self.site, dataset=self.dataset
        )

    def __len__(self) -> int:
        return len(self.refs)

    def load(self, gather_id: int) -> ShotGather:
        return load_line_gather_hardpicks(self.dataset, gather_id)
