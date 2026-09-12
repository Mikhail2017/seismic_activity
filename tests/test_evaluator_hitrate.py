"""HitRate summarization must not assign NaN into a bool Series (pandas 2)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat


def test_hitrate_summary_with_nan_errors_does_not_warn():
    ensure_hardpicks_lightning_compat()
    from hardpicks.metrics.fbp.evaluator import FBPEvaluator

    evaluator = FBPEvaluator(
        {
            "eval_metrics": [
                {"metric_type": "HitRate", "metric_params": {"buffer_size_px": 1}},
                {"metric_type": "MeanAbsoluteError"},
            ],
            "segm_class_count": 2,
            "segm_first_break_prob_threshold": 0.0,
        }
    )
    frame = pd.DataFrame({"Errors": [0.0, np.nan, 1.5, np.nan]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        metrics = evaluator._summarize_dataframe(frame)
    assert metrics["HitRate1px"] == pytest.approx(0.5)
    assert metrics["MeanAbsoluteError"] == pytest.approx(0.75)
