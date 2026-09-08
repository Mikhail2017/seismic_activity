"""Utilities for loading and reorganizing seismic first-break datasets."""

from .dataset import (
    DEFAULT_DATA_DIR,
    LineGatherRef,
    ShotGather,
    build_line_gather_index,
    ensure_hdf5_path,
    get_shot_ids,
    list_dataset_files,
    load_line_gather,
    load_shot_gather,
    load_shot_gather_from_group,
    open_trace_group,
)
from .hardpicks_bridge import (
    HardpicksGatherStore,
    hardpicks_available,
    hardpicks_item_to_shot_gather,
    open_hardpicks_dataset,
    resolve_hardpicks_site_info,
)
from .plotting import plot_shot_gather
from .sites import SITE_CONFIGS, SiteConfig, resolve_site_config

__all__ = [
    "DEFAULT_DATA_DIR",
    "HardpicksGatherStore",
    "LineGatherRef",
    "SITE_CONFIGS",
    "ShotGather",
    "SiteConfig",
    "build_line_gather_index",
    "ensure_hdf5_path",
    "get_shot_ids",
    "hardpicks_available",
    "hardpicks_item_to_shot_gather",
    "list_dataset_files",
    "load_line_gather",
    "load_shot_gather",
    "load_shot_gather_from_group",
    "open_hardpicks_dataset",
    "open_trace_group",
    "plot_shot_gather",
    "resolve_hardpicks_site_info",
    "resolve_site_config",
]
