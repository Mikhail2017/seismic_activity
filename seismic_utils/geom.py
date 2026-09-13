"""Per-trace shot-receiver geometry for GeoNorm (offset δx, relative elevation δz).

Min-max stats are fit on the training split only and reused at val/eval/predict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import numpy as np

GEOM_FIELD = "geom_features"
_EPS = 1e-6
_PAD_VALUE = 0.0


def ensure_geom_pad_field() -> None:
    """Register ``geom_features`` so collate / flip / drop keep it aligned with traces."""
    from hardpicks.data.fbp.gather_parser import ShotLineGatherDataset
    from hardpicks.data.fbp.gather_preprocess import ShotLineGatherPreprocessor

    for cls in (ShotLineGatherDataset, ShotLineGatherPreprocessor):
        fields = cls.variable_length_fields
        if not any(name == GEOM_FIELD for name, _ in fields):
            fields.append((GEOM_FIELD, _PAD_VALUE))


def geom_worker_init(_worker_id: int) -> None:
    """DataLoader worker hook: pad-field mutation is process-local under spawn."""
    ensure_geom_pad_field()


def _as_xy(coords: np.ndarray, n_traces: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.broadcast_to(arr.reshape(1, -1), (n_traces, arr.size)).copy()
    if arr.shape[0] != n_traces:
        raise ValueError(f"coords length {arr.shape[0]} != n_traces {n_traces}")
    if arr.shape[1] < 2:
        raise ValueError(f"coords must have at least x,y; got shape {arr.shape}")
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2] if arr.shape[1] > 2 else np.zeros(n_traces, dtype=np.float64)
    return x, y, z


def raw_geom_features(gather: Mapping[str, Any]) -> np.ndarray:
    """Return unnormalized ``(n_traces, 2)`` = ``[δx, δz]``.

    ``δx = hypot(sx-rx, sy-ry)`` when shot XY is present; otherwise the gather's
    shot-receiver offset. ``δz = zr - zs`` when Z is present, else 0.
    """
    rec = gather.get("rec_coords")
    if rec is None:
        raise KeyError("gather is missing rec_coords")
    rec_arr = np.asarray(rec, dtype=np.float64)
    n_traces = int(gather.get("trace_count") or rec_arr.shape[0])
    rec_x, rec_y, rec_z = _as_xy(rec_arr, n_traces)

    shot = gather.get("shot_coords")
    shot_arr = None if shot is None else np.asarray(shot, dtype=np.float64).reshape(-1)
    shot_ok = (
        shot_arr is not None
        and shot_arr.size >= 2
        and not np.allclose(shot_arr[: min(3, shot_arr.size)], 0.0)
    )
    if shot_ok:
        shot_x, shot_y, shot_z = _as_xy(np.asarray(shot, dtype=np.float64).reshape(-1), n_traces)
        dx = np.hypot(shot_x - rec_x, shot_y - rec_y)
        dz = rec_z - shot_z
    else:
        offsets = gather.get("offset_distances")
        if offsets is not None:
            dx = np.abs(np.asarray(offsets, dtype=np.float64).reshape(n_traces, -1)[:, 0])
        elif gather.get("offset") is not None:
            dx = np.abs(np.asarray(gather["offset"], dtype=np.float64).reshape(-1)[:n_traces])
        else:
            dx = np.zeros(n_traces, dtype=np.float64)
        dz = rec_z.copy()
        if np.allclose(rec_z, 0.0):
            dz = np.zeros(n_traces, dtype=np.float64)
    out = np.stack([dx, dz], axis=1).astype(np.float32, copy=False)
    return out


@dataclass(frozen=True)
class GeomStats:
    dx_min: float
    dx_max: float
    dz_min: float
    dz_max: float
    n_traces: int = 0

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> Optional["GeomStats"]:
        if not data:
            return None
        try:
            return cls(
                dx_min=float(data["dx_min"]),
                dx_max=float(data["dx_max"]),
                dz_min=float(data["dz_min"]),
                dz_max=float(data["dz_max"]),
                n_traces=int(data.get("n_traces") or 0),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _unit_range(lo: float, hi: float) -> tuple[float, float]:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi - lo < _EPS:
        return float(lo), float(lo) + 1.0
    return float(lo), float(hi)


def normalize_geom(geom: np.ndarray, stats: GeomStats) -> np.ndarray:
    arr = np.asarray(geom, dtype=np.float32)
    dx_lo, dx_hi = _unit_range(stats.dx_min, stats.dx_max)
    dz_lo, dz_hi = _unit_range(stats.dz_min, stats.dz_max)
    out = np.empty_like(arr, dtype=np.float32)
    out[:, 0] = np.clip((arr[:, 0] - dx_lo) / (dx_hi - dx_lo), 0.0, 1.0)
    out[:, 1] = np.clip((arr[:, 1] - dz_lo) / (dz_hi - dz_lo), 0.0, 1.0)
    return out


def attach_geom_features(gather: dict[str, Any], stats: GeomStats) -> dict[str, Any]:
    """Write normalized ``geom_features`` after augmentations, before collate padding."""
    ensure_geom_pad_field()
    raw = raw_geom_features(gather)
    gather[GEOM_FIELD] = normalize_geom(raw, stats)
    return gather


def accumulate_geom_stats(parser) -> GeomStats:
    """Fit min-max on unaugmented metadata (training files only)."""
    dx_min, dx_max = np.inf, -np.inf
    dz_min, dz_max = np.inf, -np.inf
    n_traces = 0
    n = len(parser)
    for i in range(n):
        meta = parser.get_meta_gather(i)
        geom = raw_geom_features(meta)
        if geom.size == 0:
            continue
        dx_min = min(dx_min, float(np.min(geom[:, 0])))
        dx_max = max(dx_max, float(np.max(geom[:, 0])))
        dz_min = min(dz_min, float(np.min(geom[:, 1])))
        dz_max = max(dz_max, float(np.max(geom[:, 1])))
        n_traces += int(geom.shape[0])
    if n_traces == 0:
        raise RuntimeError("no traces available to fit geometry min-max stats")
    dx_min, dx_max = _unit_range(dx_min, dx_max)
    dz_min, dz_max = _unit_range(dz_min, dz_max)
    return GeomStats(dx_min=dx_min, dx_max=dx_max, dz_min=dz_min, dz_max=dz_max, n_traces=n_traces)


class GeomFeatureDataset:
    """Attach normalized geometry after the inner parser (post-aug, pre-collate)."""

    def __init__(self, dataset, stats: GeomStats):
        ensure_geom_pad_field()
        self.dataset = dataset
        self.stats = stats

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        gather = self.dataset[index]
        return attach_geom_features(gather, self.stats)

    def __getitems__(self, indices):
        """PyTorch 2.x DataLoader batched fetch; must not fall through to the inner dataset."""
        return [self[int(i)] for i in indices]

    def get_meta_gather(self, gather_id: int) -> dict[str, Any]:
        return self.dataset.get_meta_gather(gather_id)

    def __getattr__(self, name: str):
        # Never forward dunders: hasattr(wrapper, "__getitems__") would otherwise
        # bind the inner parser's method and skip attach_geom_features.
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)


def collate_with_geom(batch, pad_to_nearest_pow2: bool = True, stats: GeomStats | None = None):
    """Pad-aware collate that attaches ``geom_features`` if a worker skipped the wrapper."""
    ensure_geom_pad_field()
    if stats is not None:
        for sample in batch:
            if isinstance(sample, dict) and GEOM_FIELD not in sample:
                attach_geom_features(sample, stats)
    from hardpicks.data.fbp.collate import fbp_batch_collate

    return fbp_batch_collate(batch, pad_to_nearest_pow2)


def wrap_geom_features(parser, stats: GeomStats | Mapping[str, Any] | None):
    resolved = stats if isinstance(stats, GeomStats) else GeomStats.from_mapping(stats)
    if resolved is None:
        raise ValueError("geometry stats are required when GeoNorm / geom input channels are on")
    if isinstance(parser, GeomFeatureDataset):
        return parser
    return GeomFeatureDataset(parser, resolved)


def model_needs_geom(model_or_hp: Any) -> bool:
    if isinstance(model_or_hp, Mapping):
        hp = model_or_hp
    else:
        hp = dict(getattr(model_or_hp, "hparams", {}) or {})
        if not hp:
            return bool(
                getattr(model_or_hp, "use_geonorm", False)
                or getattr(model_or_hp, "use_geom_input_channels", False)
            )
    return bool(hp.get("use_geonorm") or hp.get("use_geom_input_channels"))
