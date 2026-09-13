from __future__ import annotations

import numpy as np
import pytest

from seismic_utils.minimal_preprocess import (
    MinimalAnnotationDataset,
    apply_amplitude_preprocess,
    apply_pseudo_labels,
    ceil_to_multiple,
    strip_ground_truth,
    tracewise_zscore,
    window_half_samples,
    windowed_fb_mask,
)
from seismic_utils.minimal_split import assert_unlabeled_item_has_no_gt, gather_key


def test_window_half_samples_ms():
    assert window_half_samples(2.0) == 5
    assert window_half_samples(1.0) == 10
    assert window_half_samples(2.0, window_ms=0) == 0


def test_windowed_mask_width():
    labels = np.array([0, 20, 0], dtype=np.int32)
    mask = windowed_fb_mask(labels, n_samples=40, half_width=5)
    assert set(np.unique(mask[0]).tolist()) == {-1}
    assert int((mask[1] == 1).sum()) == 11
    assert mask[1, 15] == 1 and mask[1, 25] == 1
    assert mask[1, 14] == 0 and mask[1, 26] == 0


def test_zscore_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    x = rng.normal(3.0, 2.0, size=(4, 64)).astype(np.float32)
    z = tracewise_zscore(x)
    assert z.shape == x.shape
    np.testing.assert_allclose(z.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(z.std(axis=1), 1.0, atol=1e-5)


def test_quantize_is_finite():
    x = np.linspace(-2, 2, 32, dtype=np.float32).reshape(1, -1)
    y = apply_amplitude_preprocess(x)
    assert y.dtype == np.float32
    assert np.isfinite(y).all()
    assert y.min() >= -8.1 and y.max() <= 8.1


def test_ceil_to_multiple():
    assert ceil_to_multiple(1) == 16
    assert ceil_to_multiple(16) == 16
    assert ceil_to_multiple(17) == 32


def test_strip_ground_truth_and_dataset_unlabeled():
    item = {
        "origin": "Halfmile",
        "gather_id": 1,
        "shot_id": 2,
        "rec_line_id": 3,
        "sample_rate_ms": 2.0,
        "sample_count": 32,
        "trace_count": 4,
        "first_break_labels": np.array([5, 6, 7, 8], dtype=np.int32),
        "first_break_timestamps": np.array([10.0, 12.0, 14.0, 16.0]),
        "bad_first_breaks_mask": np.zeros(4, dtype=bool),
        "samples": np.random.default_rng(0).normal(size=(4, 32)).astype(np.float32),
        "rec_ids": np.arange(4),
    }

    class _P:
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return dict(item)

        def get_meta_gather(self, i):
            return item

    ds = MinimalAnnotationDataset(_P(), [0], mode="unlabeled")
    got = ds[0]
    assert_unlabeled_item_has_no_gt(got)
    stripped = strip_ground_truth(dict(item))
    assert_unlabeled_item_has_no_gt(stripped)


def test_pseudo_labels_window_only_survivors():
    item = {
        "samples": np.zeros((3, 40), dtype=np.float32),
        "first_break_labels": np.array([9, 9, 9], dtype=np.int32),
    }
    picks = np.array([10.0, np.nan, 12.0])
    out = apply_pseudo_labels(item, picks, half_width=5)
    assert out["first_break_labels"][0] == 10
    assert out["first_break_labels"][1] == 0
    assert int((out["segmentation_mask"][1] == 1).sum()) == 0
    assert int((out["segmentation_mask"][0] == 1).sum()) == 11
    assert gather_key({"origin": "A", "gather_id": 1, "shot_id": 2, "rec_line_id": 3})[0] == "A"


def test_minimal_collate_ceil16_and_pad_amp1():
    from seismic_utils.hardpicks_bridge import hardpicks_available
    from seismic_utils.minimal_preprocess import minimal_batch_collate

    if not hardpicks_available():
        pytest.skip("hardpicks required for collate")
    items = []
    for i in range(2):
        items.append(
            {
                "origin": "Halfmile",
                "gather_id": i,
                "shot_id": i,
                "rec_line_id": 1,
                "trace_count": 3,
                "sample_count": 20,
                "sample_rate_ms": 2.0,
                "samples": np.full((3, 20), 0.25, dtype=np.float32),
                "rec_ids": np.arange(3, dtype=np.int64),
                "first_break_labels": np.array([5, 6, 7], dtype=np.int32),
                "segmentation_mask": np.full((3, 20), -1, dtype=np.int32),
                "offset_distances": np.zeros((3, 3), dtype=np.float32),
                "bad_first_breaks_mask": np.zeros(3, dtype=bool),
            }
        )
    batch = minimal_batch_collate(items)
    samples = batch["samples"]
    assert tuple(samples.shape[-2:]) == (16, 32)
    np.testing.assert_allclose(samples[0, 3:, :].numpy(), 1.0)
    np.testing.assert_allclose(samples[0, :, 20:].numpy(), 1.0)
    rec = batch["rec_ids"]
    assert int(rec[0, 3]) == -1
    mask = batch["segmentation_mask"]
    uniq = set(np.unique(mask[0, 3:].numpy()).tolist())
    assert uniq <= {0, -1}
