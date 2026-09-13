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
from .npz_parser import (
    NpzShotLineGatherDataset,
    create_npz_parser,
    default_npz_root,
    load_npz_manifest,
    resolve_npz_asset_dir,
)
from .plotting import plot_shot_gather
from .sites import SITE_CONFIGS, SiteConfig, resolve_site_config
from .sta_lta import (
    StaLtaOptions,
    pick_first_breaks,
    pick_first_breaks_ms,
    pick_first_breaks_ms_from_shot_gather,
)
from .fb_smooth import DEFAULT_SMOOTH_THRESHOLD, fb_smooth_from_logits, fb_smooth_result
try:
    from .pick_clean import (
        DEFAULT_LATERAL_MAX_DEV,
        DEFAULT_LATERAL_WINDOW,
        apply_lateral_clean_to_frame,
        clean_picks_lateral,
    )
except ImportError:  # optional; --lateral-clean lives in pick_clean.py
    DEFAULT_LATERAL_MAX_DEV = 15
    DEFAULT_LATERAL_WINDOW = 15
    apply_lateral_clean_to_frame = None  # type: ignore[assignment]
    clean_picks_lateral = None  # type: ignore[assignment]
from .pickers import (
    ALL_PICKERS,
    HORIZON_SUFFIX,
    BEFORE_AFTER_SUFFIX,
    NN_PICKERS,
    PICKER_BEFORE_AFTER,
    PICKER_FBPUNET,
    PICKER_STA_LTA,
    PickerSpec,
    attach_smooth_evaluators,
    attach_unpicked_evaluators,
    decode_argmax_fb_unpicked,
    decode_nn_picks,
    normalize_picker,
    picker_from_hparams,
    picker_from_model,
    parse_model_suffixes,
    spec_for,
    split_before_after_model,
    split_horizon_model,
)
from .minimal_split import (
    PAPER_LABELED_COUNTS,
    collect_valid_keys,
    make_minimal_split,
)
from .minimal_preprocess import (
    apply_amplitude_preprocess,
    minimal_batch_collate,
    uses_minimal_preprocess,
    window_half_samples,
)
from .pseudo_label_qc import qc_gather_picks
from .fbp_eval_report import paper_pick_metrics

__all__ = [
    "DEFAULT_DATA_DIR",
    "HardpicksGatherStore",
    "LineGatherRef",
    "NpzShotLineGatherDataset",
    "SITE_CONFIGS",
    "ShotGather",
    "SiteConfig",
    "build_line_gather_index",
    "create_npz_parser",
    "default_npz_root",
    "ensure_hdf5_path",
    "get_shot_ids",
    "hardpicks_available",
    "hardpicks_item_to_shot_gather",
    "list_dataset_files",
    "load_line_gather",
    "load_npz_manifest",
    "load_shot_gather",
    "load_shot_gather_from_group",
    "open_hardpicks_dataset",
    "open_trace_group",
    "plot_shot_gather",
    "pick_first_breaks",
    "pick_first_breaks_ms",
    "pick_first_breaks_ms_from_shot_gather",
    "resolve_hardpicks_site_info",
    "resolve_npz_asset_dir",
    "resolve_site_config",
    "ALL_PICKERS",
    "DEFAULT_SMOOTH_THRESHOLD",
    "DEFAULT_LATERAL_MAX_DEV",
    "DEFAULT_LATERAL_WINDOW",
    "apply_lateral_clean_to_frame",
    "clean_picks_lateral",
    "BEFORE_AFTER_SUFFIX",
    "HORIZON_SUFFIX",
    "NN_PICKERS",
    "PICKER_BEFORE_AFTER",
    "PICKER_FBPUNET",
    "PICKER_STA_LTA",
    "PickerSpec",
    "StaLtaOptions",
    "PAPER_LABELED_COUNTS",
    "apply_amplitude_preprocess",
    "attach_smooth_evaluators",
    "attach_unpicked_evaluators",
    "collect_valid_keys",
    "decode_argmax_fb_unpicked",
    "decode_nn_picks",
    "make_minimal_split",
    "minimal_batch_collate",
    "paper_pick_metrics",
    "qc_gather_picks",
    "uses_minimal_preprocess",
    "window_half_samples",
    "fb_smooth_from_logits",
    "fb_smooth_result",
    "normalize_picker",
    "picker_from_hparams",
    "parse_model_suffixes",
    "picker_from_model",
    "spec_for",
    "split_before_after_model",
    "split_horizon_model",
]
