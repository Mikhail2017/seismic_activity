"""Per-asset site parameters (aligned with hardpicks / official FBP demo)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SiteConfig:
    """Asset-specific HDF5 conventions."""

    site_name: str
    hdf5_file_name: str
    first_break_field_name: str
    receiver_id_digit_count: int


SITE_CONFIGS: tuple[SiteConfig, ...] = (
    SiteConfig(
        site_name="Brunswick",
        hdf5_file_name="Brunswick_orig_1500ms_V2.hdf5",
        first_break_field_name="SPARE1",
        receiver_id_digit_count=3,
    ),
    SiteConfig(
        site_name="Halfmile",
        hdf5_file_name="Halfmile3D_add_geom_sorted.hdf5",
        first_break_field_name="SPARE1",
        receiver_id_digit_count=4,
    ),
    SiteConfig(
        site_name="Lalor",
        hdf5_file_name="Lalor_raw_z_1500ms_norp_geom_v3.hdf5",
        first_break_field_name="SPARE2",
        receiver_id_digit_count=3,
    ),
    SiteConfig(
        site_name="Sudbury",
        hdf5_file_name="preprocessed_Sudbury3D.hdf",
        first_break_field_name="SPARE1",
        receiver_id_digit_count=3,
    ),
)

_BY_FILENAME = {c.hdf5_file_name: c for c in SITE_CONFIGS}
_BY_NAME = {c.site_name.lower(): c for c in SITE_CONFIGS}


def resolve_site_config(
    path: str | Path | None = None,
    *,
    site_name: str | None = None,
    receiver_id_digit_count: int | None = None,
    first_break_field_name: str | None = None,
) -> SiteConfig:
    """
    Resolve site settings from an explicit name, HDF5 filename, or overrides.

    Falls back to SPARE1 / 3 receiver digits when the file is unknown.
    """
    base: SiteConfig | None = None
    if site_name:
        base = _BY_NAME.get(site_name.lower())
        if base is None:
            raise KeyError(f"Unknown site_name={site_name!r}; known={list(_BY_NAME)}")

    if base is None and path is not None:
        name = Path(path).name
        if name.endswith(".xz"):
            name = Path(name).stem
        base = _BY_FILENAME.get(name)

    if base is None:
        inferred = "unknown"
        if path is not None:
            stem = Path(path).name
            if stem.endswith(".xz"):
                stem = Path(stem).stem
            inferred = Path(stem).stem.split("_")[0]
        base = SiteConfig(
            site_name=inferred,
            hdf5_file_name=Path(path).name if path else "",
            first_break_field_name="SPARE1",
            receiver_id_digit_count=3,
        )

    if receiver_id_digit_count is None and first_break_field_name is None:
        return base

    return SiteConfig(
        site_name=base.site_name,
        hdf5_file_name=base.hdf5_file_name,
        first_break_field_name=first_break_field_name or base.first_break_field_name,
        receiver_id_digit_count=(
            receiver_id_digit_count
            if receiver_id_digit_count is not None
            else base.receiver_id_digit_count
        ),
    )


def receiver_line_ids(rec_peg: np.ndarray, digit_count: int) -> np.ndarray:
    """Decode receiver-line IDs from REC_PEG (hardpicks convention)."""
    pegs = np.asarray(rec_peg).reshape(-1).astype(np.int64)
    return pegs // (10**digit_count)
