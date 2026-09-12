"""Gallery selection for high-RMSE eval plots."""

import numpy as np
import pandas as pd
import pytest

from seismic_utils.fbp_eval_report import gather_summary, pick_high_rmse_gathers
from train.fbp_eval import parse_args


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
