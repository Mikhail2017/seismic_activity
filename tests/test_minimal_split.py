from __future__ import annotations

import copy

import numpy as np
import pytest

from seismic_utils.minimal_split import (
    assert_unlabeled_item_has_no_gt,
    collect_valid_keys,
    gather_key,
    is_valid_gather,
    labeled_fraction,
    make_minimal_split,
    paper_labeled_count,
    split_keys,
)


def _meta(origin, gid, shot, n=10, n_labeled=10, line=1):
    labels = np.zeros(n, dtype=np.int32)
    labels[:n_labeled] = np.arange(1, n_labeled + 1)
    bad = labels <= 0
    return {
        "origin": origin,
        "gather_id": gid,
        "shot_id": shot,
        "rec_line_id": line,
        "first_break_labels": labels,
        "bad_first_breaks_mask": bad,
    }


class _FakeParser:
    def __init__(self, metas):
        self.metas = metas

    def __len__(self):
        return len(self.metas)

    def get_meta_gather(self, i):
        return self.metas[i]


def test_labeled_fraction_and_filter():
    keep = _meta("S", 0, 1, n=100, n_labeled=5)
    drop = _meta("S", 1, 1, n=100, n_labeled=0)
    assert labeled_fraction(keep) == pytest.approx(0.05)
    assert is_valid_gather(keep)
    assert not is_valid_gather(drop)


def test_paper_counts():
    assert paper_labeled_count("Brunswick") == 148
    assert paper_labeled_count("Halfmile") == 54
    assert paper_labeled_count("Lalor") == 120
    assert paper_labeled_count("Sudbury") == 43
    with pytest.raises(KeyError):
        paper_labeled_count("Kevitsa")


def test_make_minimal_split_exact_counts_and_disjoint():
    metas = [_meta("Halfmile", i, i, n=20, n_labeled=10) for i in range(80)]
    keys = [gather_key(m) for m in metas]
    split = make_minimal_split(keys, site="Halfmile", seed=0)
    train = split_keys(split, "labeled_train")
    val = split_keys(split, "labeled_val")
    pool = split_keys(split, "unlabeled_pool")
    assert len(train) + len(val) == 54
    assert abs(len(train) / 54 - 0.75) < 0.02
    assert len(pool) == 80 - 54
    assert not set(train) & set(val)
    assert not set(train) & set(pool)
    assert not set(val) & set(pool)
    other = make_minimal_split(keys, site="Halfmile", seed=1)
    assert split_keys(other, "labeled_train") != train


def test_collect_valid_keys_drops_sparse():
    parser = _FakeParser(
        [
            _meta("S", 0, 0, n=100, n_labeled=2),
            _meta("S", 1, 1, n=100, n_labeled=0),
        ]
    )
    keys = collect_valid_keys(parser)
    assert keys == [gather_key(parser.get_meta_gather(0))]


def test_unlabeled_guard_rejects_gt():
    item = {
        "first_break_labels": np.array([0, 12, 0]),
        "segmentation_mask": np.zeros((3, 8), dtype=np.int32),
    }
    with pytest.raises(AssertionError):
        assert_unlabeled_item_has_no_gt(item)
    clean = copy.deepcopy(item)
    clean["first_break_labels"] = np.zeros(3, dtype=np.int32)
    clean["segmentation_mask"][:] = -1
    assert_unlabeled_item_has_no_gt(clean)
