"""Linear time window, far-offset rebalance, and robustness augs."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from hardpicks.data.fbp.constants import BAD_FIRST_BREAK_PICK_INDEX
from hardpicks.data.fbp.gather_transforms import generate_segmentation_mask
from hardpicks.models.constants import DONTCARE_SEGM_MASK_LABEL
from seismic_utils.gather_border import (
    apply_linear_time_window,
    kill_traces,
    rebalance_offsets,
    reverse_polarity,
    unshift_sample_indices,
)
from train.fbp_train import (
    DEFAULT_LINEAR_TIME_WINDOW,
    linear_time_window_from_hparams,
    resolve_linear_time_window,
)


def _gather(
    *,
    n_traces=8,
    n_samples=256,
    labels=None,
    offsets=None,
    rec_x=None,
    shot_x=0.0,
    sample_rate_ms=2.0,
):
    if offsets is None:
        offsets = np.linspace(100.0, 800.0, n_traces)
    if rec_x is None:
        rec_x = np.linspace(100.0, 800.0, n_traces)
    if labels is None:
        labels = np.linspace(20, 80, n_traces).astype(np.int32)
    rec_coords = np.stack(
        [np.asarray(rec_x, dtype=np.float64), np.zeros(n_traces), np.zeros(n_traces)],
        axis=1,
    )
    offset_distances = np.zeros((n_traces, 3), dtype=np.float64)
    offset_distances[:, 0] = np.abs(offsets)
    if n_traces > 1:
        rec_diff = np.abs(np.diff(rec_x))
        offset_distances[:-1, 1] = rec_diff
        offset_distances[1:, 2] = rec_diff
    samples = np.zeros((n_traces, n_samples), dtype=np.float32)
    for i, lab in enumerate(labels):
        if lab > BAD_FIRST_BREAK_PICK_INDEX:
            samples[i, int(lab)] = 1.0 + 0.1 * i
    return {
        "samples": samples,
        "first_break_labels": np.asarray(labels, dtype=np.int32),
        "first_break_timestamps": np.asarray(labels, dtype=np.float32) * sample_rate_ms,
        "bad_first_breaks_mask": np.asarray(labels) <= BAD_FIRST_BREAK_PICK_INDEX,
        "offset_distances": offset_distances,
        "rec_coords": rec_coords,
        "shot_coords": np.array([shot_x, 0.0, 0.0], dtype=np.float64),
        "trace_count": n_traces,
        "sample_count": n_samples,
        "sample_rate_ms": sample_rate_ms,
        "rec_ids": np.arange(n_traces),
    }


def test_linear_time_window_centers_two_slope_trend():
    n = 9
    rec_x = np.linspace(-400.0, 400.0, n)
    labels = (40 + 0.1 * np.abs(rec_x)).astype(np.int32)
    orig = _gather(n_traces=n, n_samples=256, labels=labels, rec_x=rec_x, offsets=rec_x, shot_x=0.0)
    gather = copy.deepcopy(orig)
    apply_linear_time_window(gather, half_window_samples=32, min_control_picks=2)
    assert gather["sample_count"] == 64
    valid = gather["first_break_labels"] > BAD_FIRST_BREAK_PICK_INDEX
    assert valid.all()
    np.testing.assert_allclose(gather["first_break_labels"][valid], 32, atol=1)
    recovered = unshift_sample_indices(
        gather["first_break_labels"], gather["sample_time_shift"]
    )
    np.testing.assert_array_equal(recovered.astype(np.int32), orig["first_break_labels"])
    for i in range(n):
        src = 32 + int(gather["sample_time_shift"][i])
        assert orig["samples"][i, src] == pytest.approx(gather["samples"][i, 32])


def test_linear_time_window_skip_when_unlabeled():
    labels = np.zeros(6, dtype=np.int32)
    gather = _gather(n_traces=6, labels=labels)
    orig_count = gather["sample_count"]
    orig_samples = gather["samples"].copy()
    apply_linear_time_window(
        gather, half_window_samples=32, unlabeled_fallback="skip", min_control_picks=2
    )
    assert gather["sample_count"] == orig_count
    np.testing.assert_array_equal(gather["samples"], orig_samples)
    np.testing.assert_array_equal(gather["sample_time_shift"], np.zeros(6, dtype=np.int32))


def test_linear_time_window_velocity_fallback_when_unlabeled():
    labels = np.zeros(6, dtype=np.int32)
    gather = _gather(n_traces=6, n_samples=512, labels=labels, offsets=np.linspace(200, 700, 6))
    apply_linear_time_window(
        gather,
        half_window_samples=64,
        unlabeled_fallback="velocity",
        fallback_velocity_mps=5000.0,
        min_control_picks=2,
    )
    assert gather["sample_count"] == 128
    assert gather["sample_time_shift"].shape == (6,)


def test_rebalance_offsets_keeps_far_traces():
    np.random.seed(0)
    offsets = np.array([50, 80, 90, 100, 600, 700, 800, 900], dtype=np.float64)
    gather = _gather(n_traces=8, offsets=offsets, rec_x=offsets, labels=np.arange(10, 18))
    rec_ids = gather["rec_ids"].copy()
    rebalance_offsets(gather, near_offset_m=200.0, drop_near_fraction=1.0)
    kept_offsets = gather["offset_distances"][:, 0]
    assert (kept_offsets >= 200.0).all()
    assert gather["trace_count"] == 4
    assert set(gather["rec_ids"]).isdisjoint(set(rec_ids[:4]))


def test_polarity_negates_all_traces_at_prob_one():
    gather = _gather()
    orig = gather["samples"].copy()
    reverse_polarity(gather, prob=1.0)
    np.testing.assert_array_equal(gather["samples"], -orig)
    np.testing.assert_array_equal(gather["first_break_labels"], _gather()["first_break_labels"])


def test_kill_invalidate_labels_are_dontcare_in_mask():
    np.random.seed(1)
    gather = _gather(n_traces=8, labels=np.arange(10, 18))
    kill_traces(gather, prob=0.99, invalidate_labels=True)
    generate_segmentation_mask(gather, segm_class_count=2)
    killed = np.isclose(gather["samples"], 0).all(axis=1)
    assert killed.any()
    for i, dead in enumerate(killed):
        if dead:
            assert gather["bad_first_breaks_mask"][i]
            assert (gather["segmentation_mask"][i] == DONTCARE_SEGM_MASK_LABEL).all()


def test_resolve_linear_time_window_defaults_and_enabled():
    missing = resolve_linear_time_window({})
    assert missing == DEFAULT_LINEAR_TIME_WINDOW
    assert missing["enabled"] is False
    disabled = resolve_linear_time_window({"linear_time_window": {"enabled": False}})
    assert disabled["enabled"] is False
    enabled = resolve_linear_time_window(
        {"linear_time_window": {"enabled": True, "half_window_samples": 256}}
    )
    assert enabled["enabled"] is True
    assert enabled["half_window_samples"] == 256
    assert enabled["unlabeled_fallback"] == "skip"
    with pytest.raises(SystemExit, match="unknown keys"):
        resolve_linear_time_window({"linear_time_window": {"enabled": True, "bogus": 1}})


def test_linear_time_window_from_hparams_only_when_enabled():
    assert linear_time_window_from_hparams({}) is None
    hp = {
        "training_data": {
            "site_params": {"linear_time_window": {"enabled": False, "half_window_samples": 32}}
        }
    }
    assert linear_time_window_from_hparams(hp) is None
    hp["training_data"]["site_params"]["linear_time_window"]["enabled"] = True
    cfg = linear_time_window_from_hparams(hp)
    assert cfg is not None and cfg["enabled"] is True


def test_valid_parser_applies_window_without_augmentations(tiny_hdf5):
    from conftest import make_parser

    parser = make_parser(
        tiny_hdf5,
        linear_time_window={
            "enabled": True,
            "half_window_samples": 32,
            "min_control_picks": 2,
            "unlabeled_fallback": "skip",
            "fallback_velocity_mps": 5500,
        },
    )
    assert parser.augmentations is None
    item = parser[0]
    assert item["sample_count"] == 64
    assert "sample_time_shift" in item
    valid = item["first_break_labels"] > BAD_FIRST_BREAK_PICK_INDEX
    np.testing.assert_allclose(item["first_break_labels"][valid], 32, atol=1)
    recovered = unshift_sample_indices(
        item["first_break_labels"], item["sample_time_shift"]
    )
    np.testing.assert_array_equal(recovered.astype(np.int32), [20, 40, 60, 80])


def test_preprocessor_dispatches_new_aug_types():
    from seismic_utils.gather_preprocess_local import LocalShotLineGatherPreprocessor

    gather = _gather(n_traces=8, offsets=np.linspace(50, 900, 8))

    class _Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, _idx):
            return copy.deepcopy(gather)

        def get_meta_gather(self, _idx):
            return copy.deepcopy(gather)

    wrapped = LocalShotLineGatherPreprocessor(
        dataset=_Dataset(),
        normalize_samples=False,
        normalize_offsets=False,
        generate_segm_masks=False,
        augmentations=[
            {
                "type": "rebalance_offsets",
                "params": {"near_offset_m": 200.0, "drop_near_fraction": 0.5},
            },
            {"type": "polarity", "params": {"prob": 0.5}},
            {"type": "kill", "params": {"prob": 0.1, "invalidate_labels": True}},
            {
                "type": "drop_and_pad",
                "params": {
                    "target_trace_counts": [8, 16],
                    "full_snap": True,
                    "drop_edges_next": False,
                },
            },
        ],
    )
    item = wrapped[0]
    assert item["trace_count"] > 0
    assert item["samples"].ndim == 2
