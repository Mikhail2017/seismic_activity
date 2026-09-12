import numpy as np
import pytest
import torch

from seismic_utils.fb_smooth import fb_smooth_from_logits, fb_smooth_result


def test_padding_is_not_a_pick():
    classes = torch.zeros(2, 1, 128, dtype=torch.long)
    classes[0, :, 80:] = 1
    classes[1, :, 30:] = 1
    logits = torch.stack((classes == 0, classes == 1), dim=1).float() * 12 - 6
    picks, probabilities = fb_smooth_from_logits(logits, threshold=7, sample_counts=[64, 64])
    assert picks.tolist() == [[0], [30]]
    assert torch.isnan(probabilities[0, 0])


def test_legacy_window_sum_rule_is_preserved():
    # Intentional compatibility behavior, not a promise of a contiguous run.
    classes = np.zeros((128, 1), dtype=np.int64)
    classes[[10, 70], 0] = 1
    assert fb_smooth_result(classes, threshold=50).tolist() == [10]


def test_invalid_sample_counts_are_rejected():
    with pytest.raises(ValueError, match="sample_counts"):
        fb_smooth_from_logits(torch.zeros(1, 2, 1, 128), sample_counts=[129])