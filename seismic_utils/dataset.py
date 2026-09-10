"""Load and reorganize HDF5 seismic datasets into shot × receiver-line gathers."""

from __future__ import annotations

import lzma
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .sites import SiteConfig, receiver_line_ids, resolve_site_config

DEFAULT_DATA_DIR = Path("/home/mika/data/seismic_activity")
TRACE_GROUP = "TRACE_DATA/DEFAULT"
DATA_KEY = "data_array"
SHOT_KEY = "SHOTID"
SHOT_PEG_KEY = "SHOT_PEG"
REC_PEG_KEY = "REC_PEG"
CHANNEL_KEY = "CHANNEL"
REC_X_KEY = "REC_X"
REC_Y_KEY = "REC_Y"
OFFSET_KEY = "OFFSET"
SAMP_RATE_KEY = "SAMP_RATE"
# Kept for older call sites; prefer SiteConfig.first_break_field_name.
FIRST_BREAK_KEY = "SPARE1"


@dataclass(frozen=True)
class LineGatherRef:
    """Lightweight handle for one shot × receiver-line gather."""

    gather_id: int
    shot_id: int
    line_id: int
    trace_indices: np.ndarray  # CHANNEL-ordered absolute HDF5 row indices

    @property
    def n_traces(self) -> int:
        return int(self.trace_indices.shape[0])

    @property
    def label(self) -> str:
        return f"{self.gather_id}: shot={self.shot_id} line={self.line_id} (n={self.n_traces})"


@dataclass(frozen=True)
class ShotGather:
    """One 2D seismic image: a shot × receiver-line gather (or a full shot)."""

    shot_id: int | float
    traces: np.ndarray  # shape (n_traces, n_samples)
    first_breaks_ms: np.ndarray  # shape (n_traces,), NaN where unlabeled
    channel: np.ndarray  # shape (n_traces,), CHANNEL order
    rec_x: np.ndarray
    rec_y: np.ndarray
    sample_rate_us: float
    offset: np.ndarray | None = None
    line_id: int | None = None
    gather_id: int | None = None

    @property
    def n_traces(self) -> int:
        return int(self.traces.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.traces.shape[1])

    @property
    def time_ms(self) -> np.ndarray:
        """Sample times in milliseconds."""
        return np.arange(self.n_samples, dtype=np.float64) * (self.sample_rate_us / 1000.0)

    @property
    def labeled_mask(self) -> np.ndarray:
        return np.isfinite(self.first_breaks_ms)

    def first_break_sample_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert first-break times (ms) to sample indices.

        Returns ``(fb_idx, mask)`` where unlabeled traces have ``fb_idx == -1``
        and ``mask == False``.
        """
        fb_idx = np.full(self.n_traces, -1, dtype=np.int64)
        mask = self.labeled_mask
        if not np.any(mask):
            return fb_idx, mask
        dt_ms = self.sample_rate_us / 1000.0
        vals = np.rint(self.first_breaks_ms[mask] / dt_ms).astype(np.int64)
        vals = np.clip(vals, 0, self.n_samples - 1)
        fb_idx[mask] = vals
        return fb_idx, mask


def list_dataset_files(data_dir: str | Path = DEFAULT_DATA_DIR) -> list[Path]:
    """List HDF5 / compressed HDF5 files under *data_dir*."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    patterns = ("*.hdf5", "*.h5", "*.hdf", "*.hdf5.xz", "*.hdf.xz", "*.h5.xz")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    # Prefer uncompressed counterparts when both exist.
    uncompressed = {p for p in files if p.suffix.lower() != ".xz"}
    filtered: list[Path] = []
    for path in files:
        if path.suffix.lower() == ".xz":
            candidate = path.with_suffix("")
            if candidate in uncompressed:
                continue
        filtered.append(path)
    return sorted(set(filtered))


def ensure_hdf5_path(path: str | Path, decompress_dir: str | Path | None = None) -> Path:
    """
    Resolve *path* to a readable HDF5 file.

    If the path points to an ``.xz`` archive, decompress it next to the archive
    (or into *decompress_dir*) when the uncompressed file is missing.
    """
    src = Path(path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Dataset not found: {src}")

    if src.suffix.lower() != ".xz":
        return src

    out_dir = Path(decompress_dir).expanduser().resolve() if decompress_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.with_suffix("").name
    if dst.exists() and dst.stat().st_size > 0:
        return dst

    with lzma.open(src, "rb") as compressed, open(dst, "wb") as out:
        shutil.copyfileobj(compressed, out, length=16 * 1024 * 1024)
    return dst


def open_trace_group(path: str | Path) -> tuple[h5py.File, h5py.Group]:
    """Open an HDF5 file and return ``(file_handle, TRACE_DATA/DEFAULT group)``."""
    hdf5_path = ensure_hdf5_path(path)
    handle = h5py.File(hdf5_path, "r")
    try:
        group = handle[TRACE_GROUP]
    except KeyError as exc:
        handle.close()
        raise KeyError(f"Missing group '{TRACE_GROUP}' in {hdf5_path}") from exc
    return handle, group


def _as_1d(dataset: Any) -> np.ndarray:
    arr = np.asarray(dataset)
    return arr.reshape(-1)


def _read_rows(dataset: Any, indices: np.ndarray) -> np.ndarray:
    """
    Read rows at arbitrary *indices*.

    h5py requires strictly increasing fancy indices, so we sort for the read
    and restore the caller's order afterwards.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return np.asarray(dataset[0:0])

    sort_for_read = np.argsort(indices, kind="mergesort")
    read_idx = indices[sort_for_read]
    values = np.asarray(dataset[read_idx])
    restore = np.argsort(sort_for_read, kind="mergesort")
    return values[restore]


def _normalize_first_breaks(values: np.ndarray) -> np.ndarray:
    """Map unlabeled markers (0 / -1) to NaN; keep other values in milliseconds."""
    fb = values.astype(np.float64, copy=True)
    unlabeled = (fb == 0) | (fb == -1)
    fb[unlabeled] = np.nan
    return fb


def _order_by_channel(channel: np.ndarray) -> np.ndarray:
    """Sort traces by acquisition CHANNEL (stable mergesort)."""
    return np.argsort(np.asarray(channel).reshape(-1), kind="mergesort")


def _shot_id_array(group: h5py.Group) -> np.ndarray:
    """Prefer SHOTID; fall back to SHOT_PEG when needed."""
    if SHOT_KEY in group:
        shots = _as_1d(group[SHOT_KEY][()]).astype(np.int64, copy=False)
        if np.unique(shots).size > 1 or SHOT_PEG_KEY not in group:
            return shots
    if SHOT_PEG_KEY in group:
        return _as_1d(group[SHOT_PEG_KEY][()]).astype(np.int64, copy=False)
    raise KeyError(f"Missing '{SHOT_KEY}' / '{SHOT_PEG_KEY}' in {TRACE_GROUP}")


def get_shot_ids(path: str | Path) -> list[int | float]:
    """Return sorted unique SHOTID values for the dataset."""
    handle, group = open_trace_group(path)
    try:
        shot_ids = _shot_id_array(group)
        unique = np.unique(shot_ids)
        return [int(v) for v in unique]
    finally:
        handle.close()


def build_line_gather_index(
    path: str | Path,
    *,
    site: SiteConfig | None = None,
    site_name: str | None = None,
    receiver_id_digit_count: int | None = None,
) -> list[LineGatherRef]:
    """
    Build shot × receiver-line gather index (hardpicks / task splitting).

    Receiver lines are decoded from ``REC_PEG // 10**digit_count``. Each gather
    is the intersection of one shot and one receiver line, ordered by CHANNEL.
    """
    site_cfg = site or resolve_site_config(
        path,
        site_name=site_name,
        receiver_id_digit_count=receiver_id_digit_count,
    )
    handle, group = open_trace_group(path)
    try:
        return build_line_gather_index_from_group(group, site_cfg)
    finally:
        handle.close()


def build_line_gather_index_from_group(
    group: h5py.Group,
    site: SiteConfig,
) -> list[LineGatherRef]:
    """Build gather index from an already-open HDF5 group."""
    required = (DATA_KEY, REC_PEG_KEY, CHANNEL_KEY)
    missing = [key for key in required if key not in group]
    if missing:
        raise KeyError(f"Missing keys in {TRACE_GROUP}: {missing}")

    shot_ids = _shot_id_array(group)
    line_ids = receiver_line_ids(_as_1d(group[REC_PEG_KEY][()]), site.receiver_id_digit_count)
    channels = _as_1d(group[CHANNEL_KEY][()]).astype(np.int64, copy=False)

    # Pack (shot, line) into a single key for grouping.
    line_u = line_ids.astype(np.uint64, copy=False)
    shot_u = shot_ids.astype(np.uint64, copy=False)
    # line IDs fit in ~32 bits for these assets; leave headroom.
    pair_key = (shot_u << 32) | (line_u & np.uint64(0xFFFFFFFF))

    order = np.lexsort((channels, pair_key))
    pair_sorted = pair_key[order]
    boundaries = np.flatnonzero(np.r_[True, pair_sorted[1:] != pair_sorted[:-1], True])

    gathers: list[LineGatherRef] = []
    for gather_id, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        idx = order[start:stop]
        gathers.append(
            LineGatherRef(
                gather_id=gather_id,
                shot_id=int(shot_ids[idx[0]]),
                line_id=int(line_ids[idx[0]]),
                trace_indices=idx.astype(np.int64, copy=False),
            )
        )
    return gathers


def _load_gather_from_indices(
    group: h5py.Group,
    ordered_idx: np.ndarray,
    *,
    shot_id: int | float,
    first_break_field: str,
    line_id: int | None = None,
    gather_id: int | None = None,
) -> ShotGather:
    if first_break_field not in group:
        raise KeyError(f"Missing first-break field '{first_break_field}' in {TRACE_GROUP}")
    required = (DATA_KEY, CHANNEL_KEY, REC_X_KEY, REC_Y_KEY)
    missing = [key for key in required if key not in group]
    if missing:
        raise KeyError(f"Missing keys in {TRACE_GROUP}: {missing}")

    traces = np.asarray(_read_rows(group[DATA_KEY], ordered_idx), dtype=np.float32)
    first_breaks = _normalize_first_breaks(_as_1d(_read_rows(group[first_break_field], ordered_idx)))
    channel = _as_1d(_read_rows(group[CHANNEL_KEY], ordered_idx)).astype(np.int64, copy=False)
    rec_x = _as_1d(_read_rows(group[REC_X_KEY], ordered_idx)).astype(np.float64, copy=False)
    rec_y = _as_1d(_read_rows(group[REC_Y_KEY], ordered_idx)).astype(np.float64, copy=False)
    # Match hardpicks: apply abs(COORD_SCALE) to XY / OFFSET (SEG-Y style integer coords).
    xy_scale = 1.0
    if "COORD_SCALE" in group:
        raw_scale = float(_as_1d(_read_rows(group["COORD_SCALE"], ordered_idx[:1]))[0])
        if raw_scale != 0.0:
            xy_scale = abs(raw_scale)
    rec_x = rec_x / xy_scale
    rec_y = rec_y / xy_scale
    offset = None
    if OFFSET_KEY in group:
        offset = _as_1d(_read_rows(group[OFFSET_KEY], ordered_idx)).astype(np.float64, copy=False)
        offset = offset / xy_scale
    if SAMP_RATE_KEY in group:
        sample_rate = float(_as_1d(_read_rows(group[SAMP_RATE_KEY], ordered_idx[:1]))[0])
    else:
        sample_rate = 1000.0

    return ShotGather(
        shot_id=shot_id,
        traces=traces,
        first_breaks_ms=first_breaks,
        channel=channel,
        rec_x=rec_x,
        rec_y=rec_y,
        sample_rate_us=sample_rate,
        offset=offset,
        line_id=line_id,
        gather_id=gather_id,
    )


def load_line_gather_from_ref(
    group: h5py.Group,
    ref: LineGatherRef,
    *,
    site: SiteConfig,
) -> ShotGather:
    """Load amplitude + labels for a previously indexed line gather."""
    return _load_gather_from_indices(
        group,
        ref.trace_indices,
        shot_id=ref.shot_id,
        first_break_field=site.first_break_field_name,
        line_id=ref.line_id,
        gather_id=ref.gather_id,
    )


def load_line_gather(
    path: str | Path,
    *,
    gather_id: int | None = None,
    shot_id: int | None = None,
    line_id: int | None = None,
    site: SiteConfig | None = None,
    site_name: str | None = None,
    index: list[LineGatherRef] | None = None,
) -> ShotGather:
    """
    Load one shot × receiver-line gather.

    Identify the gather either by ``gather_id`` or by ``(shot_id, line_id)``.
    Pass a precomputed ``index`` to avoid rebuilding it on every call.
    """
    site_cfg = site or resolve_site_config(path, site_name=site_name)
    handle, group = open_trace_group(path)
    try:
        gather_index = index if index is not None else build_line_gather_index_from_group(group, site_cfg)
        ref: LineGatherRef | None = None
        if gather_id is not None:
            if 0 <= gather_id < len(gather_index) and gather_index[gather_id].gather_id == gather_id:
                ref = gather_index[gather_id]
            else:
                for item in gather_index:
                    if item.gather_id == gather_id:
                        ref = item
                        break
            if ref is None:
                raise ValueError(f"gather_id={gather_id} not found")
        elif shot_id is not None and line_id is not None:
            for item in gather_index:
                if item.shot_id == shot_id and item.line_id == line_id:
                    ref = item
                    break
            if ref is None:
                raise ValueError(f"No gather for shot={shot_id} line={line_id}")
        else:
            raise ValueError("Provide gather_id or both shot_id and line_id")
        return load_line_gather_from_ref(group, ref, site=site_cfg)
    finally:
        handle.close()


def load_shot_gather_from_group(
    group: h5py.Group,
    all_shot_ids: np.ndarray,
    shot_id: int | float,
    *,
    first_break_field: str = FIRST_BREAK_KEY,
) -> ShotGather:
    """Load one full-SHOTID gather (all receiver lines), CHANNEL-ordered."""
    indices = np.flatnonzero(all_shot_ids == shot_id)
    if indices.size == 0:
        raise ValueError(f"SHOTID {shot_id} not found")
    channel = _as_1d(_read_rows(group[CHANNEL_KEY], indices))
    order = _order_by_channel(channel)
    ordered_idx = indices[order]
    return _load_gather_from_indices(
        group,
        ordered_idx,
        shot_id=shot_id,
        first_break_field=first_break_field,
    )


def load_shot_gather(path: str | Path, shot_id: int | float) -> ShotGather:
    """
    Load a single full-SHOTID gather, ordered by CHANNEL.

    Prefer ``load_line_gather`` / ``build_line_gather_index`` for the task's
    shot × receiver-line images.
    """
    site = resolve_site_config(path)
    handle, group = open_trace_group(path)
    try:
        all_shot_ids = _shot_id_array(group)
        return load_shot_gather_from_group(
            group,
            all_shot_ids,
            shot_id,
            first_break_field=site.first_break_field_name,
        )
    finally:
        handle.close()
