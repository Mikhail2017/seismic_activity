"""HDF5 parser with copy-on-read metadata ownership.

Hardpicks transforms are intentionally in-place. Never hand them arrays owned
by the raw metadata cache (or by the raw dataset's geometry/index maps).
"""

import copy

import numpy as np

from hardpicks.data.fbp.gather_parser import ShotLineGatherDataset


class OwnedMetadataGatherDataset(ShotLineGatherDataset):
    """Keep raw metadata cached, but give every caller an independent record."""

    def get_meta_gather(self, gather_id):
        return copy.deepcopy(super().get_meta_gather(gather_id))


def create_hdf5_parser(*, site_info, site_params, prefix, dataset_hyper_params, segm_class_count):
    """Local equivalent of FBPDataModule.create_parser with safe raw metadata."""
    from hardpicks.data.fbp.gather_cleaner import ShotLineGatherCleaner
    from hardpicks.data.fbp.gather_preprocess import ShotLineGatherPreprocessor
    from hardpicks.data.fbp.gather_splitter import get_train_and_test_sub_datasets

    params = copy.deepcopy(site_params)
    parser = OwnedMetadataGatherDataset(
        hdf5_path=site_info["processed_hdf5_path"],
        site_name=site_info["site_name"],
        receiver_id_digit_count=site_info["receiver_id_digit_count"],
        first_break_field_name=site_info["first_break_field_name"],
        **dataset_hyper_params,
    )
    if prefix in {"valid", "test", "predict"}:
        if params.get("auto_fill_missing_picks") or params.get("augmentations") or params.get("segm_first_break_buffer"):
            raise ValueError("Evaluation cannot fill labels, augment gathers, or buffer first-break masks")
    cleaner_keys = (
        "auto_invalidate_outlier_picks", "outlier_detection_strategy",
        "outlier_detection_filter_size", "outlier_detection_threshold",
        "auto_fill_missing_picks", "pick_fill_strategy", "pick_fill_max_dist",
        "rejected_gather_yaml_path",
    )
    parser = ShotLineGatherCleaner(parser, **{key: params.get(key) for key in cleaner_keys})
    preprocess_keys = (
        "normalize_samples", "sample_norm_strategy", "normalize_offsets",
        "shot_to_rec_offset_norm_const", "rec_to_rec_offset_norm_const",
        "generate_first_break_prior_masks", "first_break_prior_velocity_range",
        "first_break_prior_offset_range", "segm_first_break_buffer", "augmentations",
        "linear_time_window",
    )
    parser = ShotLineGatherPreprocessor(
        parser, segm_class_count=segm_class_count,
        generate_segm_masks=params.get("generate_segm_masks", bool(segm_class_count)),
        **{key: params.get(key) for key in preprocess_keys},
    )
    if "subset" in params:
        subset = params["subset"]
        ratio = subset["eval_ratio"]
        if not 0 < ratio < 1:
            raise ValueError("eval_ratio must be between zero and one")
        train, valid = get_train_and_test_sub_datasets(
            shot_line_gather_dataset=parser,
            random_number_generator=np.random.default_rng(subset.get("split_seed", 0)),
            fraction_of_shots_in_testing_set=ratio,
            fraction_of_lines_in_testing_set=ratio,
            ignore_line_ids_if_unique=subset.get("ignore_line_ids", False),
        )
        parser = valid if subset["use_eval_split"] else train
    return parser