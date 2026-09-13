from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from seismic_utils.fbp_eval_report import paper_pick_metrics
from seismic_utils.pickers import decode_argmax_fb_unpicked, decode_nn_picks


def test_decode_unpicked_when_background_wins():
    logits = torch.zeros(1, 2, 3, 8)
    logits[0, 0] = 2.0
    logits[0, 1] = 0.0
    picks, _ = decode_argmax_fb_unpicked(logits)
    assert torch.equal(picks[0], torch.zeros(3, dtype=picks.dtype))


def test_decode_argmax_fb_channel():
    logits = torch.zeros(1, 2, 2, 6)
    logits[0, 0] = 0.0
    logits[0, 1, 0, 4] = 5.0
    logits[0, 1, 1, 1] = 5.0
    picks, _ = decode_argmax_fb_unpicked(logits)
    assert int(picks[0, 0]) == 4
    assert int(picks[0, 1]) == 1


def test_decode_nn_picks_uses_unpicked_recipe():
    class _M:
        hparams = {"minimal_annotations": {"ablation": "combined"}}

    logits = torch.zeros(1, 2, 1, 4)
    logits[0, 0] = 3.0
    picks, _ = decode_nn_picks(logits, _M())
    assert int(picks[0, 0]) == 0


def test_paper_metrics_missing_pred_is_total_error():
    df = pd.DataFrame(
        {
            "Predictions": [10, 0, 12],
            "Errors": [0.0, 11.0, 1.0],
            "AbsError": [0.0, 11.0, 1.0],
        }
    )
    metrics = paper_pick_metrics(df)
    assert metrics["n_labeled"] == 3
    assert metrics["Coverage"] == pytest.approx(2 / 3)
    assert metrics["W_pred_0"] == pytest.approx(0.5)
    assert metrics["W_total_0"] == pytest.approx(1 / 3)
    assert metrics["W_total_10"] == pytest.approx(2 / 3)
    assert metrics["MAE"] == pytest.approx(0.5)
    assert np.isfinite(metrics["W_pred_10"])


def test_crossentropy_weight_list_becomes_tensor():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "models" / "losses.py"
    spec = importlib.util.spec_from_file_location("_meneses_losses", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    loss = mod.get_loss_function("crossentropy", None, {"weight": [1, 100]}, ignore_index=-1)
    assert torch.equal(loss.weight, torch.tensor([1.0, 100.0]))
    assert loss.ignore_index == -1


def test_leakyrelu_opt_in_in_unet_base():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "models" / "unet_base.py").read_text()
    assert "LeakyReLU(negative_slope=0.01" in text
    assert 'activation: typing.Optional' in text
    assert 'hyper_params.get("activation", "relu")' in text
