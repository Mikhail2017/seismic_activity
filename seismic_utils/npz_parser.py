"""Fast hardpicks-compatible gather parser backed by per-gather NPZ files.

Expected layout (from ``seismic_utils.export_npz``)::

    <npz_root>/<asset>/manifest.csv
    <npz_root>/<asset>/gather_00000_shot_..._line_....npz
    <npz_root>/<asset>/meta.json

Each NPZ stores ``image`` as (n_samples, n_traces). This module converts to the
hardpicks convention ``samples`` (n_traces, n_samples) so
``ShotLineGatherPreprocessor`` / ``fbp_batch_collate`` / ``FBPUNet`` work unchanged.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import DEFAULT_DATA_DIR

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # export / viewer envs do not need torch
    _TorchDataset = object

logger = logging.getLogger(__name__)

# Match hardpicks.data.fbp.constants without requiring hardpicks at import time.
_BAD_FB_LABEL = -1
_BAD_OR_PADDED_ID = -1
_DEAD_TRACE_EPS = 1e-8


def default_npz_root(data_dir: str | Path | None = None) -> Path:
    """Return ``<data_dir>/npz``."""
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    return root / "npz"


def resolve_npz_asset_dir(
    asset: str,
    npz_root: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Resolve ``<npz_root>/<asset>`` and require it exists."""
    root = Path(npz_root) if npz_root is not None else default_npz_root(data_dir)
    asset_dir = root / asset
    if not asset_dir.is_dir():
        raise FileNotFoundError(
            f"NPZ asset directory not found: {asset_dir}\n"
            f"Export first: python -m seismic_utils.export_npz ... -o {root}"
        )
    return asset_dir


def load_npz_manifest(
    asset_dir: str | Path,
    *,
    require_labeled: bool = True,
) -> list[dict[str, Any]]:
    """
    Load ``manifest.csv`` rows (or rebuild from ``gather_*.npz`` glob).

    Paths in the manifest are relative to the NPZ root (parent of *asset_dir*).
    """
    asset_dir = Path(asset_dir)
    npz_root = asset_dir.parent
    manifest_path = asset_dir / "manifest.csv"
    rows: list[dict[str, Any]] = []

    if manifest_path.is_file():
        with manifest_path.open(newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "path": npz_root / row["path"],
                        "asset": row.get("asset", asset_dir.name),
                        "gather_id": int(row["gather_id"]),
                        "shot_id": int(row["shot_id"]),
                        "line_id": int(row["line_id"]),
                        "n_traces": int(row["n_traces"]),
                        "n_samples": int(row["n_samples"]),
                        "n_labeled": int(row["n_labeled"]),
                        "sample_rate_us": float(row["sample_rate_us"]),
                    }
                )
    else:
        logger.warning("No manifest.csv under %s; scanning gather_*.npz", asset_dir)
        for path in sorted(asset_dir.glob("gather_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                mask = np.asarray(data["mask"])
                rows.append(
                    {
                        "path": path,
                        "asset": str(np.asarray(data["asset"])) if "asset" in data.files else asset_dir.name,
                        "gather_id": int(data["gather_id"]),
                        "shot_id": int(data["shot_id"]),
                        "line_id": int(data["line_id"]),
                        "n_traces": int(data["image"].shape[1]),
                        "n_samples": int(data["image"].shape[0]),
                        "n_labeled": int(mask.sum()),
                        "sample_rate_us": float(data["sample_rate_us"]),
                    }
                )

    if require_labeled:
        before = len(rows)
        rows = [r for r in rows if r["n_labeled"] > 0]
        dropped = before - len(rows)
        if dropped:
            logger.info("Dropped %d fully-unlabeled NPZ gathers", dropped)

    if not rows:
        raise FileNotFoundError(f"No usable NPZ gathers under {asset_dir}")
    return rows


def _offset_distances_from_npz(
    n_traces: int,
    offset: np.ndarray | None,
    rec_coords: np.ndarray,
) -> np.ndarray:
    """Build hardpicks-style (n_traces, 3) offsets: shot-rec, next-rec, prev-rec."""
    out = np.zeros((n_traces, 3), dtype=np.float32)
    if offset is not None:
        out[:, 0] = np.asarray(offset, dtype=np.float32).reshape(-1)
    elif rec_coords.shape[0] == n_traces:
        # No absolute shot coords in NPZ — leave shot-rec as 0; still fill rec-rec.
        pass
    if n_traces > 1:
        diffs = np.linalg.norm(np.diff(rec_coords[:, :2], axis=0), axis=1).astype(np.float32)
        out[:-1, 1] = diffs  # dist to next
        out[1:, 2] = diffs  # dist to prev
    return out


def npz_file_to_hardpicks_gather(
    path: str | Path,
    *,
    origin: str | None = None,
    provide_offset_dists: bool = True,
) -> dict[str, Any]:
    """Load one NPZ and return a hardpicks-style gather dict (including ``samples``)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        image = np.asarray(data["image"])  # (n_samples, n_traces)
        if image.ndim != 2:
            raise ValueError(f"{path}: image must be 2D, got {image.shape}")
        samples = np.ascontiguousarray(image.T, dtype=np.float32)  # (n_traces, n_samples)
        n_traces, n_samples = samples.shape

        labeled = np.asarray(data["mask"]).reshape(-1).astype(bool)
        if labeled.shape[0] != n_traces:
            raise ValueError(f"{path}: mask length {labeled.shape[0]} != n_traces {n_traces}")

        fb_idx = np.asarray(data["fb_idx"]).reshape(-1).astype(np.int64)
        fb_labels = fb_idx.copy()
        fb_labels[~labeled] = _BAD_FB_LABEL

        if "fb_ms" in data.files:
            fb_ts = np.asarray(data["fb_ms"], dtype=np.float32).reshape(-1)
        else:
            sample_rate_us = float(data["sample_rate_us"])
            fb_ts = fb_labels.astype(np.float32) * (sample_rate_us / 1000.0)
        fb_ts = fb_ts.copy()
        fb_ts[~labeled] = float(_BAD_FB_LABEL)

        if "channel" in data.files:
            rec_ids = np.asarray(data["channel"], dtype=np.int64).reshape(-1)
        else:
            rec_ids = np.arange(n_traces, dtype=np.int64)

        rec_x = np.asarray(data["rec_x"], dtype=np.float64).reshape(-1) if "rec_x" in data.files else np.zeros(n_traces)
        rec_y = np.asarray(data["rec_y"], dtype=np.float64).reshape(-1) if "rec_y" in data.files else np.zeros(n_traces)
        rec_coords = np.stack([rec_x, rec_y, np.zeros(n_traces, dtype=np.float64)], axis=1)

        offset = np.asarray(data["offset"], dtype=np.float64).reshape(-1) if "offset" in data.files else None
        sample_rate_us = float(data["sample_rate_us"])
        asset = origin or (
            str(np.asarray(data["asset"])) if "asset" in data.files else path.parent.name
        )

        gather: dict[str, Any] = {
            "origin": asset,
            "shot_id": int(data["shot_id"]),
            "rec_line_id": int(data["line_id"]),
            "rec_ids": rec_ids,
            "gather_id": int(data["gather_id"]),
            "gather_trace_ids": np.arange(n_traces, dtype=np.int64),
            "first_break_labels": fb_labels,
            "first_break_timestamps": fb_ts,
            "bad_first_breaks_mask": ~labeled,
            "rec_coords": rec_coords,
            "shot_coords": np.zeros(3, dtype=np.float64),  # not stored in NPZ
            "trace_count": n_traces,
            "sample_count": n_samples,
            "sample_rate_ms": sample_rate_us / 1000.0,
            "dead_rec_mask": np.isclose(samples, 0, atol=_DEAD_TRACE_EPS).all(axis=1),
            "samples": samples,
        }
        if provide_offset_dists:
            gather["offset_distances"] = _offset_distances_from_npz(
                n_traces, offset, rec_coords
            )
        else:
            gather["offset_distances"] = None
    return gather


def _meta_from_npz_file(
    path: str | Path,
    *,
    origin: str,
    provide_offset_dists: bool,
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load NPZ metadata without reading the full ``image`` array when possible."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        # Prefer manifest sizes to skip touching image when present.
        if record is not None:
            n_traces = int(record["n_traces"])
            n_samples = int(record["n_samples"])
            sample_rate_us = float(record["sample_rate_us"])
        else:
            image_shape = data["image"].shape
            n_samples, n_traces = int(image_shape[0]), int(image_shape[1])
            sample_rate_us = float(data["sample_rate_us"])

        labeled = np.asarray(data["mask"]).reshape(-1).astype(bool)
        fb_idx = np.asarray(data["fb_idx"]).reshape(-1).astype(np.int64)
        fb_labels = fb_idx.copy()
        fb_labels[~labeled] = _BAD_FB_LABEL

        if "fb_ms" in data.files:
            fb_ts = np.asarray(data["fb_ms"], dtype=np.float32).reshape(-1).copy()
        else:
            fb_ts = fb_labels.astype(np.float32) * (sample_rate_us / 1000.0)
        fb_ts[~labeled] = float(_BAD_FB_LABEL)

        if "channel" in data.files:
            rec_ids = np.asarray(data["channel"], dtype=np.int64).reshape(-1)
        else:
            rec_ids = np.arange(n_traces, dtype=np.int64)

        rec_x = (
            np.asarray(data["rec_x"], dtype=np.float64).reshape(-1)
            if "rec_x" in data.files
            else np.zeros(n_traces)
        )
        rec_y = (
            np.asarray(data["rec_y"], dtype=np.float64).reshape(-1)
            if "rec_y" in data.files
            else np.zeros(n_traces)
        )
        rec_coords = np.stack([rec_x, rec_y, np.zeros(n_traces, dtype=np.float64)], axis=1)
        offset = (
            np.asarray(data["offset"], dtype=np.float64).reshape(-1)
            if "offset" in data.files
            else None
        )

        meta: dict[str, Any] = {
            "origin": origin,
            "shot_id": int(data["shot_id"]),
            "rec_line_id": int(data["line_id"]),
            "rec_ids": rec_ids,
            "gather_id": int(data["gather_id"]),
            "gather_trace_ids": np.arange(n_traces, dtype=np.int64),
            "first_break_labels": fb_labels,
            "first_break_timestamps": fb_ts,
            "bad_first_breaks_mask": ~labeled,
            "rec_coords": rec_coords,
            "shot_coords": np.zeros(3, dtype=np.float64),
            "trace_count": n_traces,
            "sample_count": n_samples,
            "sample_rate_ms": sample_rate_us / 1000.0,
        }
        if provide_offset_dists:
            meta["offset_distances"] = _offset_distances_from_npz(
                n_traces, offset, rec_coords
            )
        else:
            meta["offset_distances"] = None
    return meta


class NpzShotLineGatherDataset(_TorchDataset):
    """
    PyTorch dataset over exported line-gather NPZ files.

    Implements the hardpicks ``ShotLineGatherDatasetBase`` contract
    (``__len__``, ``__getitem__``, ``get_meta_gather``) so it can be wrapped by
    ``ShotLineGatherPreprocessor`` and fed to ``fbp_batch_collate``.
    """

    # Same pad fields as hardpicks ShotLineGatherDataset (for documentation / local collate).
    variable_length_fields = [
        ("rec_ids", _BAD_OR_PADDED_ID),
        ("gather_trace_ids", _BAD_OR_PADDED_ID),
        ("first_break_labels", 0),
        ("first_break_timestamps", 0),
        ("bad_first_breaks_mask", True),
        ("dead_rec_mask", True),
        ("rec_coords", 0),
        ("offset_distances", 0),
        ("samples", 0),
    ]

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        site_name: str | None = None,
        provide_offset_dists: bool = True,
        mmap: bool = False,
    ):
        if not records:
            raise ValueError("NpzShotLineGatherDataset requires at least one record")
        self.records = list(records)
        self.site_name = site_name or str(self.records[0].get("asset", "unknown"))
        self.provide_offset_dists = provide_offset_dists
        self.mmap = mmap
        self._meta_cache: dict[int, dict[str, Any]] = {}

    @classmethod
    def from_asset_dir(
        cls,
        asset_dir: str | Path,
        *,
        site_name: str | None = None,
        require_labeled: bool = True,
        provide_offset_dists: bool = True,
        mmap: bool = False,
    ) -> "NpzShotLineGatherDataset":
        records = load_npz_manifest(asset_dir, require_labeled=require_labeled)
        return cls(
            records,
            site_name=site_name or Path(asset_dir).name,
            provide_offset_dists=provide_offset_dists,
            mmap=mmap,
        )

    @classmethod
    def from_npz_root(
        cls,
        asset: str,
        npz_root: str | Path | None = None,
        data_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> "NpzShotLineGatherDataset":
        asset_dir = resolve_npz_asset_dir(asset, npz_root=npz_root, data_dir=data_dir)
        return cls.from_asset_dir(asset_dir, site_name=asset, **kwargs)

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        return npz_file_to_hardpicks_gather(
            rec["path"],
            origin=self.site_name,
            provide_offset_dists=self.provide_offset_dists,
        )

    def get_meta_gather(self, gather_id: int) -> dict[str, Any]:
        """Metadata without loading waveform samples (cached)."""
        if gather_id in self._meta_cache:
            return self._meta_cache[gather_id]
        rec = self.records[gather_id]
        meta = _meta_from_npz_file(
            rec["path"],
            origin=self.site_name,
            provide_offset_dists=self.provide_offset_dists,
            record=rec,
        )
        self._meta_cache[gather_id] = meta
        return meta

    def __getitem__(self, gather_id: int) -> dict[str, Any]:
        return self._load(gather_id)


def create_npz_parser(
    asset: str,
    *,
    npz_root: str | Path | None = None,
    data_dir: str | Path | None = None,
    prefix: str = "train",
    site_params: dict[str, Any] | None = None,
    segm_class_count: int = 1,
    provide_offset_dists: bool = True,
    require_labeled: bool = True,
) -> Any:
    """
    Build a training/eval dataset from NPZ gathers (hardpicks preprocessor + optional split).

    Mirrors ``FBPDataModule.create_parser`` but skips HDF5 + ``ShotLineGatherCleaner``
    (rejects / unlabeled gathers should already be handled at export or via
    ``require_labeled``).

    Parameters
    ----------
    asset:
        Subdirectory name under the NPZ root (e.g. ``\"Lalor\"``).
    prefix:
        ``\"train\"`` / ``\"valid\"`` / ``\"test\"`` — gates augmentations and buffers.
    site_params:
        Same knobs as hardpicks site params: ``normalize_samples``, ``augmentations``,
        ``subset`` (``eval_ratio``, ``use_eval_split``, optional ``split_seed``),
        ``segm_first_break_buffer``, etc.
    """
    from hardpicks.data.fbp.gather_splitter import get_train_and_test_sub_datasets
    from seismic_utils.gather_preprocess_local import wrap_gather_preprocessor

    site_params = dict(site_params or {})
    dataset = NpzShotLineGatherDataset.from_npz_root(
        asset,
        npz_root=npz_root,
        data_dir=data_dir,
        require_labeled=require_labeled,
        provide_offset_dists=provide_offset_dists,
    )

    default_generate_segm_masks = bool(segm_class_count)
    generate_segm_masks = site_params.get("generate_segm_masks", default_generate_segm_masks)
    segm_first_break_buffer = site_params.get("segm_first_break_buffer", None)
    assert not segm_first_break_buffer or prefix not in ["valid", "test"], (
        "segmentation mask first break buffer should *never* be activated in validation/testing!"
    )

    augmentations = site_params.get("augmentations", None)
    if isinstance(augmentations, dict):
        augmentations = list(augmentations.values())
    assert not augmentations or prefix not in ["valid", "test", "predict"], (
        "augmentations should *never* be activated in validation/testing!"
    )

    parser = wrap_gather_preprocessor(
        dataset,
        site_params=site_params,
        extra_kwargs=dict(
            normalize_samples=site_params.get("normalize_samples", None),
            sample_norm_strategy=site_params.get("sample_norm_strategy", None),
            normalize_offsets=site_params.get("normalize_offsets", None),
            shot_to_rec_offset_norm_const=site_params.get("shot_to_rec_offset_norm_const", None),
            rec_to_rec_offset_norm_const=site_params.get("rec_to_rec_offset_norm_const", None),
            generate_first_break_prior_masks=site_params.get("generate_first_break_prior_masks", None),
            first_break_prior_velocity_range=site_params.get("first_break_prior_velocity_range", None),
            first_break_prior_offset_range=site_params.get("first_break_prior_offset_range", None),
            generate_segm_masks=generate_segm_masks,
            segm_class_count=segm_class_count,
            segm_first_break_buffer=segm_first_break_buffer,
            augmentations=augmentations,
        ),
    )

    if "subset" in site_params:
        expected = ["eval_ratio", "use_eval_split"]
        assert all(k in site_params["subset"] for k in expected)
        split_rng = np.random.default_rng(seed=site_params["subset"].get("split_seed", 0))
        eval_ratio = site_params["subset"]["eval_ratio"]
        assert 0 < eval_ratio < 1
        parser_train, parser_eval = get_train_and_test_sub_datasets(
            shot_line_gather_dataset=parser,
            random_number_generator=split_rng,
            fraction_of_shots_in_testing_set=eval_ratio,
            fraction_of_lines_in_testing_set=eval_ratio,
            ignore_line_ids_if_unique=site_params["subset"].get("ignore_line_ids", False),
        )
        parser = parser_eval if site_params["subset"]["use_eval_split"] else parser_train

    return parser


def load_npz_meta(asset_dir: str | Path) -> dict[str, Any] | None:
    """Return ``meta.json`` contents if present."""
    path = Path(asset_dir) / "meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())
