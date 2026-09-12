"""Gallery selection for high-RMSE eval plots."""

import numpy as np
import pandas as pd
import pytest

from seismic_utils.fbp_eval_report import gather_summary, pick_high_rmse_gathers
from train.fbp_eval import (
    parse_args,
    resolve_eval_linear_time_window,
    shot_gather_for_eval_plot,
)


def test_gather_summary_includes_rmse():
    df = pd.DataFrame(
        {
            "OriginId": [0, 0],
            "GatherId": [1, 1],
            "ShotId": [10, 10],
            "Origin": ["Halfmile", "Halfmile"],
            "Errors": [3.0, -4.0],
            "AbsError": [3.0, 4.0],
        }
    )
    out = gather_summary(df)
    assert len(out) == 1
    assert out["RMSE"].iloc[0] == pytest.approx(np.sqrt((9.0 + 16.0) / 2.0))


def test_pick_high_rmse_gathers_strictly_above():
    df = pd.DataFrame(
        {
            "OriginId": [0, 0, 0],
            "GatherId": [1, 2, 3],
            "ShotId": [1, 1, 1],
            "RMSE": [5.0, 5.1, np.nan],
            "MAE": [2.0, 4.0, 3.0],
        }
    )
    out = pick_high_rmse_gathers(df, 5.0)
    assert list(out["GatherId"]) == [2]


def test_pick_high_rmse_gathers_empty_without_column():
    df = pd.DataFrame({"GatherId": [1], "MAE": [1.0]})
    out = pick_high_rmse_gathers(df, 0.0)
    assert out.empty


def test_parse_args_rmse_above():
    args = parse_args(["--ckpt", "model.ckpt", "--rmse-above", "7.5"])
    assert args.rmse_above == pytest.approx(7.5)
    assert parse_args(["--ckpt", "model.ckpt"]).rmse_above is None
    with pytest.raises(SystemExit):
        parse_args(["--ckpt", "model.ckpt", "--rmse-above", "nan"])


def test_eval_plot_uses_windowed_labels_not_original_timestamps():
    n, t = 4, 64
    item = {
        "samples": np.zeros((n, t), np.float32),
        "first_break_timestamps": np.array([400.0, 800.0, 1200.0, 1600.0]),
        "first_break_labels": np.array([32, 32, 32, 32], dtype=np.int32),
        "bad_first_breaks_mask": np.zeros(n, dtype=bool),
        "shot_id": 1,
        "gather_id": 2,
        "rec_line_id": 3,
        "sample_rate_ms": 2.0,
        "rec_coords": np.zeros((n, 2), dtype=np.float64),
        "offset_distances": np.zeros((n, 3), dtype=np.float64),
    }
    gather = shot_gather_for_eval_plot(item)
    np.testing.assert_allclose(gather.first_breaks_ms, np.full(n, 64.0))


def test_resolve_eval_linear_time_window_from_recipe(tmp_path):
    ckpt = tmp_path / "best.ckpt"
    ckpt.write_bytes(b"")
    (tmp_path / "train_recipe.yaml").write_text(
        "linear_time_window:\n  enabled: true\n  half_window_samples: 512\n"
        "  min_control_picks: 2\n  unlabeled_fallback: skip\n"
        "  fallback_velocity_mps: 5500\n"
    )
    cfg = resolve_eval_linear_time_window({}, ckpt=ckpt, ckpt_dir=tmp_path)
    assert cfg is not None
    assert cfg["enabled"] is True
    assert cfg["half_window_samples"] == 512
