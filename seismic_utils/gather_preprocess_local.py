"""Local preprocessor that works with pip-installed hardpicks.

Adds ``linear_time_window`` and extra aug types without requiring a patched
``ShotLineGatherPreprocessor.__init__``.
"""

from __future__ import annotations

import copy
import functools
from typing import Any, Dict, Optional

from hardpicks.data.fbp import gather_transforms as hp_transforms
from hardpicks.data.fbp.gather_preprocess import ShotLineGatherPreprocessor
from hardpicks.data.transforms import stochastic_op_wrapper

from seismic_utils import gather_border


def ensure_sample_time_shift_pad_field() -> None:
    """Register ``sample_time_shift`` so collate/drop/flip/pad keep it."""
    fields = list(ShotLineGatherPreprocessor.variable_length_fields)
    if any(name == "sample_time_shift" for name, _ in fields):
        return
    fields.append(("sample_time_shift", 0))
    ShotLineGatherPreprocessor.variable_length_fields = fields


class LinearTimeWindowDataset:
    """Apply the linear time window before hardpicks preprocess/augs."""

    def __init__(self, dataset, config: Dict[str, Any]):
        self.dataset = dataset
        params = dict(config)
        params.pop("enabled", None)
        self._params = params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, gather_id):
        gather = self.dataset[gather_id]
        gather_border.apply_linear_time_window(gather, **self._params)
        return gather

    def get_meta_gather(self, gather_id):
        return self.dataset.get_meta_gather(gather_id)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class LocalShotLineGatherPreprocessor(ShotLineGatherPreprocessor):
    """Upstream preprocessor plus polarity / offset rebalance / kill-invalidate."""

    supported_augmentation_strategies = list(
        ShotLineGatherPreprocessor.supported_augmentation_strategies
    ) + ["polarity", "rebalance_offsets"]

    def _get_augmentation_ops(self, augmentation_config):
        assert isinstance(augmentation_config, list)
        assert all(isinstance(a, dict) for a in augmentation_config)
        aug_ops = []
        for aug_cfg in augmentation_config:
            aug_cfg = copy.deepcopy(aug_cfg)
            assert aug_cfg["type"] in self.supported_augmentation_strategies
            if aug_cfg["type"] == "flip":
                aug_ops.append(stochastic_op_wrapper(hp_transforms.flip, 0.5))
                continue
            if aug_cfg["type"] == "crop":
                fn = self._augment_crop_samples
            elif aug_cfg["type"] == "resample_hardcoded":
                fn = self._augment_resample_hardcoded
            elif aug_cfg["type"] == "resample_nearby":
                fn = self._augment_resample_nearby
            elif aug_cfg["type"] == "drop_and_pad":
                fn = self._augment_drop_and_pad_traces
            elif aug_cfg["type"] == "kill":
                fn = gather_border.kill_traces
            elif aug_cfg["type"] == "noise":
                fn = hp_transforms.add_noise_patch
            elif aug_cfg["type"] == "polarity":
                fn = gather_border.reverse_polarity
            elif aug_cfg["type"] == "rebalance_offsets":
                fn = gather_border.rebalance_offsets
            else:
                raise NotImplementedError(aug_cfg["type"])
            params = aug_cfg.get("params") or {}
            aug_ops.append(functools.partial(fn, **params))
        return aug_ops

    @staticmethod
    def _augment_drop_and_pad_traces(
        gather,
        target_trace_counts,
        full_snap,
        max_drop_ratio=0.25,
        drop_edges_next=True,
    ):
        import numpy as np

        curr_trace_count = gather["trace_count"]
        assert len(target_trace_counts) > 0
        target_trace_counts = np.sort(np.asarray(target_trace_counts))
        target_trace_count_idx = np.argmin(np.abs(target_trace_counts - curr_trace_count))
        target_trace_count = target_trace_counts[target_trace_count_idx]
        trace_count_var = target_trace_count - curr_trace_count
        max_drop_count = int(round(max_drop_ratio * curr_trace_count))
        if trace_count_var < 0 and abs(trace_count_var) > max_drop_count:
            assert target_trace_count_idx < len(target_trace_counts) - 1, (
                f"gather too big for the current max limit in 'drop-and-pad'"
                f"(curr={curr_trace_count}, limit={target_trace_counts[-1]})"
            )
            target_trace_count = target_trace_counts[target_trace_count_idx + 1]
            trace_count_var = target_trace_count - curr_trace_count
        if trace_count_var < 0:
            assert abs(trace_count_var) <= max_drop_count
            if not full_snap:
                trace_count_var = np.random.randint(abs(trace_count_var) + 1)
            hp_transforms.drop_traces(gather, abs(trace_count_var), True, drop_edges_next)
        elif trace_count_var > 0:
            if not full_snap:
                trace_count_var = np.random.randint(trace_count_var + 1)
            prepad_size = np.random.randint(trace_count_var)
            postpad_size = trace_count_var - prepad_size
            hp_transforms.pad_traces(gather, prepad_size, postpad_size)

    def _generate_first_break_prior_masks(self, gather):
        gather_border.generate_windowed_prior_mask(
            gather,
            self.first_break_prior_velocity_range,
            self.first_break_prior_offset_range,
        )


def wrap_gather_preprocessor(
    dataset,
    *,
    site_params: Dict[str, Any],
    extra_kwargs: Optional[Dict[str, Any]] = None,
):
    """Build the local preprocessor, applying the time window first when enabled."""
    ensure_sample_time_shift_pad_field()
    params = dict(site_params or {})
    window = params.pop("linear_time_window", None)
    if isinstance(window, dict) and window.get("enabled"):
        dataset = LinearTimeWindowDataset(dataset, window)
    kwargs = dict(extra_kwargs or {})
    return LocalShotLineGatherPreprocessor(dataset, **kwargs)
