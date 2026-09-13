from __future__ import annotations

import numpy as np

from seismic_utils.pseudo_label_qc import draw_without_replacement, qc_gather_picks


def test_qc_keeps_linear_offset_picks():
    off = np.linspace(100.0, 200.0, 40)
    picks = 10.0 + 0.2 * (off - 100.0)
    out = qc_gather_picks(off, picks)
    assert out["admit"]
    assert out["n_survive"] == 40


def test_qc_rejects_gather_with_many_outliers():
    # One wild pick in each 6-trace offset group (~16.7% rejected) → below 85%.
    off = np.repeat(np.linspace(0.0, 100.0, 20), 6)
    picks = np.full(120, 50.0)
    picks.reshape(20, 6)[:, 0] = 5000.0
    out = qc_gather_picks(off, picks, n_bins=20, n_sigma=2.0, min_survive_frac=0.85)
    assert not out["admit"]
    assert np.all(np.isnan(out["picks"]))


def test_qc_single_outlier_on_linear_line():
    off = np.repeat(np.linspace(0.0, 100.0, 20), 10)
    picks = np.full(200, 30.0)
    picks[25] = 400.0
    out = qc_gather_picks(off, picks)
    assert out["admit"]
    assert not out["survive"][25]
    assert np.isnan(out["picks"][25])
    assert np.isfinite(out["picks"][0])


def test_qc_missing_pick_does_not_survive():
    off = np.linspace(0.0, 10.0, 10)
    picks = np.linspace(5.0, 15.0, 10)
    picks[3] = np.nan
    out = qc_gather_picks(off, picks, min_survive_frac=0.5)
    assert not out["survive"][3]


def test_qc_tiny_bin_no_sigma_reject():
    # Isolated offset in its own bin (n=1): 2σ must not drop the wild pick.
    off = np.array([0.0, 50.0, 50.1, 50.2, 50.3])
    picks = np.array([999.0, 10.0, 10.1, 10.2, 10.3])
    out = qc_gather_picks(off, picks, n_bins=2, min_survive_frac=0.0)
    assert out["survive"][0]
    assert out["n_traces"] == 5


def test_qc_zero_sigma_keeps():
    off = np.linspace(0.0, 10.0, 12)
    picks = np.full(12, 5.0)
    out = qc_gather_picks(off, picks)
    assert out["admit"]
    assert out["n_survive"] == 12


def test_qc_missing_offset_does_not_survive():
    off = np.linspace(0.0, 10.0, 10)
    off[2] = np.nan
    picks = np.linspace(5.0, 15.0, 10)
    out = qc_gather_picks(off, picks, min_survive_frac=0.0)
    assert not out["survive"][2]


def test_draw_without_replacement_removes_all_drawn():
    rng = np.random.default_rng(0)
    remaining = list(range(10))
    drawn, rest = draw_without_replacement(remaining, 3, rng)
    assert len(drawn) == 3
    assert len(rest) == 7
    assert set(drawn).isdisjoint(rest)
    assert set(drawn) | set(rest) == set(remaining)
