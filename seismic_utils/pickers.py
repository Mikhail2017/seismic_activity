"""First-break picker registry (FB-pixel UNet, before/after UNet, STA-LTA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .fb_smooth import DEFAULT_SMOOTH_THRESHOLD, fb_smooth_from_logits

PICKER_FBPUNET = "fbpunet"
PICKER_BEFORE_AFTER = "before_after"
PICKER_STA_LTA = "sta-lta"

NN_PICKERS = (PICKER_FBPUNET, PICKER_BEFORE_AFTER)
ALL_PICKERS = (PICKER_FBPUNET, PICKER_BEFORE_AFTER, PICKER_STA_LTA)

HORIZON_SUFFIX = "-horizon"  # train: first-break prior channel, not this picker
BEFORE_AFTER_SUFFIX = "-before-after"


@dataclass(frozen=True)
class PickerSpec:
    name: str
    segm_class_count: int | None
    needs_ckpt: bool
    decoder: str


SPECS: dict[str, PickerSpec] = {
    PICKER_FBPUNET: PickerSpec(PICKER_FBPUNET, 1, True, "argmax_fb"),
    PICKER_BEFORE_AFTER: PickerSpec(PICKER_BEFORE_AFTER, 2, True, "fb_smooth"),
    PICKER_STA_LTA: PickerSpec(PICKER_STA_LTA, None, False, "sta_lta"),
}

_ALIASES = {
    "unet": PICKER_FBPUNET,
    "model": PICKER_FBPUNET,
    "ckpt": PICKER_FBPUNET,
    "checkpoint": PICKER_FBPUNET,
    "fb-pixel": PICKER_FBPUNET,
    "fb_pixel": PICKER_FBPUNET,
    "horizon": PICKER_BEFORE_AFTER,
    "before-after": PICKER_BEFORE_AFTER,
    "before_after": PICKER_BEFORE_AFTER,
    "stalta": PICKER_STA_LTA,
    "sta/lta": PICKER_STA_LTA,
    "sta-lta-os": PICKER_STA_LTA,
}


def normalize_picker(name: str | None) -> str:
    mode = (name or PICKER_FBPUNET).strip().lower().replace("_", "-")
    mode = _ALIASES.get(mode, mode)
    if mode not in SPECS:
        known = ", ".join(ALL_PICKERS)
        raise ValueError(f"Unknown picker {name!r}; expected one of: {known}")
    return mode


def spec_for(name: str | None) -> PickerSpec:
    return SPECS[normalize_picker(name)]


def parse_model_suffixes(name: str) -> tuple[str, bool, bool]:
    """Split encoder + optional ``-before-after`` / ``-horizon`` (any order).

    ``resnet18-before-after-horizon`` → ``('resnet18', True, True)``.
    ``-horizon`` is the first-break prior channel in train, not a picker.
    """
    raw = (name or "").strip()
    if not raw:
        return raw, False, False
    key = raw.lower().replace("_", "-")
    is_ba = False
    is_horizon = False
    while True:
        if key.endswith(HORIZON_SUFFIX):
            base = key[: -len(HORIZON_SUFFIX)].rstrip("-")
            if not base:
                raise ValueError("'-horizon' needs an encoder preset, e.g. resnet18-horizon")
            key = base
            is_horizon = True
            continue
        if key.endswith(BEFORE_AFTER_SUFFIX):
            base = key[: -len(BEFORE_AFTER_SUFFIX)].rstrip("-")
            if not base:
                raise ValueError(
                    "'-before-after' needs an encoder preset, e.g. resnet18-before-after"
                )
            key = base
            is_ba = True
            continue
        break
    return key, is_ba, is_horizon


def split_before_after_model(name: str) -> tuple[str, bool]:
    """Split ``resnet34-before-after`` → ``(resnet34, True)``.

    ``-horizon`` is the first-break prior channel in train, not this picker.
    Combined labels keep the prior suffix on the returned encoder key so
    ``resnet18-before-after-horizon`` → ``('resnet18-horizon', True)``.
    """
    base, is_ba, is_horizon = parse_model_suffixes(name)
    if is_horizon:
        return f"{base}{HORIZON_SUFFIX}", is_ba
    return base, is_ba


def split_horizon_model(name: str) -> tuple[str, str]:
    """Split ``resnet18-before-after-horizon`` → ``(resnet18, before_after)``."""
    base, is_ba, _ = parse_model_suffixes(name)
    return base, PICKER_BEFORE_AFTER if is_ba else PICKER_FBPUNET


def picker_from_hparams(hp: Mapping[str, Any] | None) -> str:
    data = dict(hp or {})
    explicit = data.get("picker")
    if explicit:
        try:
            return normalize_picker(str(explicit))
        except ValueError:
            pass
    sc = data.get("segm_class_count")
    try:
        if int(sc) == 2:
            return PICKER_BEFORE_AFTER
    except (TypeError, ValueError):
        pass
    return PICKER_FBPUNET


def picker_from_model(model) -> str:
    hp = dict(getattr(model, "hparams", {}) or {})
    return picker_from_hparams(hp)


def reconcile_cli_picker(cli_picker: str, ckpt_picker: str) -> tuple[str, bool]:
    """Prefer checkpoint picker when it disagrees with a neural-net CLI value.

    Returns ``(resolved_picker, mismatched)``.
    """
    cli = normalize_picker(cli_picker)
    ckpt = normalize_picker(ckpt_picker)
    if cli == PICKER_STA_LTA:
        return cli, False
    if cli != ckpt:
        return ckpt, True
    return cli, False


def smooth_threshold_from_hparams(hp: Mapping[str, Any] | None) -> int:
    data = dict(hp or {})
    raw = data.get("segm_first_break_smooth_threshold", DEFAULT_SMOOTH_THRESHOLD)
    try:
        return max(int(raw), 1)
    except (TypeError, ValueError):
        return DEFAULT_SMOOTH_THRESHOLD


def decode_nn_picks(raw_preds, model, *, picker: str | None = None, smooth_threshold: int | None = None):
    """Logits → ``(pick_indices, probabilities)`` for a neural picker."""
    resolved = normalize_picker(picker or picker_from_model(model))
    if resolved == PICKER_BEFORE_AFTER:
        hp = dict(getattr(model, "hparams", {}) or {})
        thr = int(smooth_threshold) if smooth_threshold is not None else smooth_threshold_from_hparams(hp)
        return fb_smooth_from_logits(raw_preds, threshold=thr)

    import hardpicks.metrics.fbp.utils as metrics_utils

    evaluator = getattr(model, "test_evaluator", None)
    scheme = int(getattr(evaluator, "segm_class_count", None) or getattr(model, "segm_class_count", 1))
    thresh = float(getattr(evaluator, "segm_first_break_prob_threshold", 0.0) or 0.0)
    return metrics_utils.get_regr_preds_from_raw_preds(
        raw_preds=raw_preds,
        segm_class_count=scheme,
        prob_threshold=thresh,
    )


class SmoothFBPEvaluator:
    """``FBPEvaluator`` whose decode uses :func:`fb_smooth_from_logits`.

    Instantiated lazily so importing this module does not require hardpicks.
    """

    def __new__(cls, hyper_params: Mapping[str, Any]):
        return _smooth_evaluator_class()(dict(hyper_params))


def _smooth_evaluator_class():
    cached = getattr(_smooth_evaluator_class, "_cached", None)
    if cached is not None:
        return cached

    import numpy as np
    import pandas as pd
    import torch

    import hardpicks.metrics.fbp.utils as utils
    from hardpicks.metrics.fbp.evaluator import (
        FBPEvaluator,
        SUPPORTED_REGRESSION_METRICS,
        SUPPORTED_SEGMENTATION_METRICS,
    )

    class _SmoothFBPEvaluator(FBPEvaluator):
        def __init__(self, hyper_params):
            super().__init__(hyper_params)
            self.smooth_threshold = smooth_threshold_from_hparams(hyper_params)

        def ingest(self, batch, batch_idx, raw_preds):
            if not self.metrics_metamap:
                return {}
            assert batch_idx not in self.seen_batch_idxs, "we've seen this minibatch already!"
            assert len(batch["rec_ids"]) == len(batch["offset_distances"])

            regr_preds, probabilities_of_fbp = fb_smooth_from_logits(
                raw_preds, threshold=self.smooth_threshold, sample_counts=batch.get("sample_count")
            )
            if self.extract_fbp_probability:
                assert probabilities_of_fbp is not None, "probabilities_of_fbp is None"

            good_traces_per_gather, trace_rec_ids, trace_origin_ids = [], [], []
            trace_shot_ids, trace_gather_ids, trace_offsets, trace_preds, trace_probs = [], [], [], [], []
            batched_receiver_ids = batch["rec_ids"].cpu().numpy()
            batched_offset_distances = batch["offset_distances"].cpu().numpy()
            for gather_idx in range(batch["batch_size"]):
                receiver_ids = batched_receiver_ids[gather_idx]
                good_traces_mask = utils.get_valid_traces_mask(receiver_ids)
                good_trace_idxs = np.where(good_traces_mask)[0]
                trace_count = len(good_trace_idxs)
                good_traces_per_gather.append(good_trace_idxs)
                trace_rec_ids.extend(receiver_ids[good_trace_idxs])
                gather_origin_name = batch["origin"][gather_idx]
                if gather_origin_name not in self.origin_id_map:
                    self.origin_id_map[gather_origin_name] = len(self.origin_id_map)
                trace_origin_ids.extend([self.origin_id_map[gather_origin_name]] * trace_count)
                trace_shot_ids.extend([int(batch["shot_id"][gather_idx])] * trace_count)
                trace_gather_ids.extend([int(batch["gather_id"][gather_idx])] * trace_count)
                offset_distances = batched_offset_distances[gather_idx, good_trace_idxs, 0]
                trace_offsets.extend(offset_distances)
                trace_preds.extend(regr_preds[gather_idx, good_trace_idxs].cpu().numpy())
                if self.extract_fbp_probability:
                    trace_probs.extend(probabilities_of_fbp[gather_idx, good_trace_idxs].cpu().numpy())

            tot_trace_count = sum(len(idxs) for idxs in good_traces_per_gather)
            self.seen_batch_idxs[batch_idx] = (
                self.accumulated_trace_counts,
                self.accumulated_trace_counts + tot_trace_count,
            )
            curr_dataframe = {
                "GatherId": pd.Series(data=trace_gather_ids, dtype="int"),
                "ShotId": pd.Series(data=trace_shot_ids, dtype="int"),
                "ReceiverId": pd.Series(data=trace_rec_ids, dtype="int"),
                "OriginId": pd.Series(data=trace_origin_ids, dtype="int"),
                "Offset": pd.Series(data=trace_offsets, dtype="float"),
                "Predictions": pd.Series(data=trace_preds, dtype="int"),
            }
            if self.extract_fbp_probability:
                curr_dataframe["Probabilities"] = pd.Series(data=trace_probs, dtype="float")

            use_segm_eval = any(
                m[0] in SUPPORTED_SEGMENTATION_METRICS for m in self.metrics_metamap.values()
            )
            if use_segm_eval:
                eval_arrays = self._get_segm_metrics_arrays(
                    batch, regr_preds, good_traces_per_gather
                )
                for col_name, array in eval_arrays.items():
                    assert col_name not in curr_dataframe
                    assert len(array) == tot_trace_count
                    curr_dataframe[col_name] = pd.Series(data=array, dtype=None)
            use_regr_eval = any(
                m[0] in SUPPORTED_REGRESSION_METRICS for m in self.metrics_metamap.values()
            )
            if use_regr_eval:
                error_array = self._get_regr_error_array(batch, regr_preds, good_traces_per_gather)
                assert len(error_array) == tot_trace_count
                curr_dataframe["Errors"] = pd.Series(data=error_array, dtype="float")
            curr_dataframe = pd.DataFrame(curr_dataframe)
            assert np.array_equal(self._dataframe.columns, curr_dataframe.columns)
            self.accumulated_trace_counts += len(curr_dataframe)
            self.list_batch_dataframes.append(curr_dataframe)
            if len(self.list_batch_dataframes) > 500:
                self.finalize()
            return self._summarize_dataframe(curr_dataframe)

    _smooth_evaluator_class._cached = _SmoothFBPEvaluator  # type: ignore[attr-defined]
    return _SmoothFBPEvaluator


def attach_smooth_evaluators(model, hyper_params: Mapping[str, Any]) -> None:
    """Replace Lightning evaluators so valid/HitRate uses ``fb_smooth_result``."""
    from hardpicks.metrics.base import NoneEvaluator

    hp = dict(hyper_params)
    keep_train_none = isinstance(model.train_evaluator, NoneEvaluator)
    cls = _smooth_evaluator_class()
    model.valid_evaluator = cls(hp)
    model.test_evaluator = cls(hp)
    model.pred_evaluator = cls(hp)
    if not keep_train_none:
        model.train_evaluator = cls(hp)


def make_eval_evaluator(hyper_params: Mapping[str, Any], picker: str):
    """Evaluator used by ``fbp_eval.py`` for a neural picker."""
    hp = dict(hyper_params)
    if normalize_picker(picker) == PICKER_BEFORE_AFTER:
        return _smooth_evaluator_class()(hp)
    from hardpicks.metrics.fbp.evaluator import FBPEvaluator

    return FBPEvaluator(hp)
