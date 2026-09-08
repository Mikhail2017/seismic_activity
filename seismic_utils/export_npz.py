"""Convert HDF5 seismic assets into per line-gather NPZ files for training/eval."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .dataset import (
    DEFAULT_DATA_DIR,
    ShotGather,
    build_line_gather_index_from_group,
    load_line_gather_from_ref,
    open_trace_group,
)
from .hardpicks_bridge import HardpicksGatherStore, hardpicks_available
from .sites import resolve_site_config


def infer_asset_name(hdf5_path: str | Path, asset: str | None = None) -> str:
    """Infer a short asset name from the HDF5 filename unless *asset* is given."""
    if asset:
        return asset
    site = resolve_site_config(hdf5_path)
    if site.site_name != "unknown":
        return site.site_name
    stem = Path(hdf5_path).name
    if stem.endswith(".xz"):
        stem = Path(stem).stem
    stem = Path(stem).stem
    token = re.split(r"[_\-]", stem, maxsplit=1)[0]
    return token or stem


def gather_to_npz_dict(gather: ShotGather, *, asset: str) -> dict[str, np.ndarray]:
    """
    Build the NPZ payload for one line gather.

    Arrays
    ------
    image : float32, shape (H, W)
        Amplitudes with time along axis 0 and traces along axis 1 (gather order).
    fb_idx : int64, shape (W,)
        First-break sample indices; ``-1`` where unlabeled.
    mask : bool, shape (W,)
        ``True`` for labeled traces.
    shot_id, line_id, gather_id : int64 scalars
    sample_rate_us : float32 scalar
    asset : unicode scalar
    fb_ms, channel, rec_x, rec_y [, offset] : per-trace metadata
    """
    fb_idx, mask = gather.first_break_sample_indices()
    if gather.line_id is None or gather.gather_id is None:
        raise ValueError("Line-gather NPZ export requires line_id and gather_id")

    payload: dict[str, np.ndarray] = {
        "image": np.ascontiguousarray(gather.traces.T, dtype=np.float32),
        "fb_idx": fb_idx,
        "mask": mask.astype(bool, copy=False),
        "shot_id": np.asarray(gather.shot_id, dtype=np.int64),
        "line_id": np.asarray(gather.line_id, dtype=np.int64),
        "gather_id": np.asarray(gather.gather_id, dtype=np.int64),
        "sample_rate_us": np.asarray(gather.sample_rate_us, dtype=np.float32),
        "asset": np.asarray(asset),
        "fb_ms": gather.first_breaks_ms.astype(np.float32, copy=False),
        "channel": gather.channel.astype(np.int64, copy=False),
        "rec_x": gather.rec_x.astype(np.float64, copy=False),
        "rec_y": gather.rec_y.astype(np.float64, copy=False),
    }
    if gather.offset is not None:
        payload["offset"] = gather.offset.astype(np.float64, copy=False)
    return payload


def npz_path_for_gather(
    output_dir: str | Path,
    asset: str,
    gather_id: int,
    shot_id: int,
    line_id: int,
) -> Path:
    """Return ``{output_dir}/{asset}/gather_{id:05d}_shot_{shot}_line_{line}.npz``."""
    name = f"gather_{gather_id:05d}_shot_{int(shot_id)}_line_{int(line_id)}.npz"
    return Path(output_dir) / asset / name


def save_gather_npz(
    gather: ShotGather,
    output_path: str | Path,
    *,
    asset: str,
    compress: bool = False,
) -> Path:
    """Write one gather NPZ to *output_path*."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gather_to_npz_dict(gather, asset=asset)
    saver = np.savez_compressed if compress else np.savez
    saver(path, **payload)
    return path


def export_hdf5_to_npz(
    hdf5_path: str | Path,
    output_dir: str | Path,
    *,
    asset: str | None = None,
    compress: bool = False,
    skip_existing: bool = True,
    write_manifest: bool = True,
    limit: int | None = None,
    site_name: str | None = None,
    backend: str = "hardpicks",
) -> list[Path]:
    """
    Convert an HDF5 seismic asset into one NPZ file per shot × receiver-line gather.

    Default ``backend="hardpicks"`` uses the official ``ShotLineGatherDataset`` so
    gather IDs match training. Pass ``backend="native"`` for the local splitter.
    """
    hdf5_path = Path(hdf5_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    site = resolve_site_config(hdf5_path, site_name=site_name)
    asset_name = asset or site.site_name
    asset_dir = output_dir / asset_name
    asset_dir.mkdir(parents=True, exist_ok=True)

    backend = (backend or "hardpicks").strip().lower()
    if backend == "auto":
        backend = "hardpicks"

    use_hardpicks = backend == "hardpicks"
    if use_hardpicks and not hardpicks_available():
        raise RuntimeError(
            "Default backend is hardpicks, but it is not importable "
            "(need hardpicks + torch). Install deps or pass --backend native."
        )
    if backend not in {"hardpicks", "native"}:
        raise ValueError(f"Unknown backend={backend!r}; use hardpicks or native")

    written: list[Path] = []
    split_backend = "hardpicks" if use_hardpicks else "native"

    if use_hardpicks:
        store = HardpicksGatherStore(hdf5_path, site=site)
        refs = store.refs
        if limit is not None:
            refs = refs[:limit]
        for ref in tqdm(refs, desc=f"Export {asset_name}", unit="gather"):
            out_path = npz_path_for_gather(
                output_dir, asset_name, ref.gather_id, ref.shot_id, ref.line_id
            )
            if skip_existing and out_path.exists():
                written.append(out_path)
                continue
            gather = store.load(ref.gather_id)
            save_gather_npz(gather, out_path, asset=asset_name, compress=compress)
            written.append(out_path)
    else:
        handle, group = open_trace_group(hdf5_path)
        try:
            index = build_line_gather_index_from_group(group, site)
            if limit is not None:
                index = index[:limit]
            for ref in tqdm(index, desc=f"Export {asset_name}", unit="gather"):
                out_path = npz_path_for_gather(
                    output_dir, asset_name, ref.gather_id, ref.shot_id, ref.line_id
                )
                if skip_existing and out_path.exists():
                    written.append(out_path)
                    continue
                gather = load_line_gather_from_ref(group, ref, site=site)
                save_gather_npz(gather, out_path, asset=asset_name, compress=compress)
                written.append(out_path)
            refs = index
        finally:
            handle.close()

    if write_manifest:
        _rebuild_manifest(output_dir, asset_dir, asset_name)
        meta = {
            "asset": asset_name,
            "source_hdf5": str(Path(hdf5_path).resolve()),
            "site_name": site.site_name,
            "first_break_field_name": site.first_break_field_name,
            "receiver_id_digit_count": site.receiver_id_digit_count,
            "n_gathers": len(refs),
            "compress": compress,
            "split": "shot_x_receiver_line",
            "backend": split_backend,
            "keys": [
                "image",
                "fb_idx",
                "mask",
                "shot_id",
                "line_id",
                "gather_id",
                "sample_rate_us",
                "asset",
                "fb_ms",
                "channel",
                "rec_x",
                "rec_y",
                "offset",
            ],
        }
        (asset_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    return written


def _rebuild_manifest(output_dir: Path, asset_dir: Path, asset_name: str) -> None:
    rows: list[dict[str, object]] = []
    for path in sorted(asset_dir.glob("gather_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            mask = np.asarray(data["mask"])
            rows.append(
                {
                    "path": str(path.relative_to(output_dir)),
                    "asset": asset_name,
                    "gather_id": int(data["gather_id"]),
                    "shot_id": int(data["shot_id"]),
                    "line_id": int(data["line_id"]),
                    "n_traces": int(data["image"].shape[1]),
                    "n_samples": int(data["image"].shape[0]),
                    "n_labeled": int(mask.sum()),
                    "sample_rate_us": float(data["sample_rate_us"]),
                }
            )

    manifest_path = asset_dir / "manifest.csv"
    fieldnames = [
        "path",
        "asset",
        "gather_id",
        "shot_id",
        "line_id",
        "n_traces",
        "n_samples",
        "n_labeled",
        "sample_rate_us",
    ]
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export HDF5 seismic data to per shot×receiver-line NPZ files.",
    )
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        default=str(DEFAULT_DATA_DIR / "Brunswick_orig_1500ms_V2.hdf5"),
        help="Path to HDF5 (or .hdf5.xz) file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_DATA_DIR / "npz"),
        help="Output root directory (default: <data_dir>/npz)",
    )
    parser.add_argument("--asset", default=None, help="Asset name stored in NPZ files")
    parser.add_argument("--site-name", default=None, help="Override site config name")
    parser.add_argument(
        "--backend",
        choices=("hardpicks", "native", "auto"),
        default="hardpicks",
        help="Gather splitter (default: hardpicks)",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Use np.savez_compressed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export only the first N line gathers",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    paths = export_hdf5_to_npz(
        args.hdf5_path,
        args.output_dir,
        asset=args.asset,
        compress=args.compress,
        skip_existing=not args.overwrite,
        limit=args.limit,
        site_name=args.site_name,
        backend=args.backend,
    )
    print(f"Exported {len(paths)} NPZ files under {args.output_dir}")


if __name__ == "__main__":
    main()
