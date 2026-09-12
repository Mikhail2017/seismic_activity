"""GeoNorm modulation, geometry stats, and train-recipe ablation flags."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat

ensure_hardpicks_lightning_compat()

from models.geonorm import (
    GeoNorm,
    GeomEncoder,
    geom_embedding_context,
    group_count,
    replace_norms_with_geonorm,
)
from seismic_utils.geom import (
    GeomStats,
    attach_geom_features,
    normalize_geom,
    raw_geom_features,
)
from train.fbp_train import resolve_geonorm_config


def test_geonorm_zero_init_matches_groupnorm():
    layer = GeoNorm(16, geom_dim=8, groups=8)
    x = torch.randn(2, 16, 12, 20)
    emb = torch.randn(2, 12, 8)
    gn = nn.GroupNorm(8, 16, affine=False)
    with geom_embedding_context(emb):
        out = layer(x)
    torch.testing.assert_close(out, gn(x), atol=1e-5, rtol=1e-5)


def test_geonorm_modulates_traces_not_time():
    layer = GeoNorm(4, geom_dim=6, groups=2)
    nn.init.ones_(layer.to_mod.weight)
    nn.init.zeros_(layer.to_mod.bias)
    x = torch.ones(1, 4, 8, 10)
    emb = torch.zeros(1, 8, 6)
    emb[0, 0] = 1.0
    emb[0, 3] = -1.0
    with geom_embedding_context(emb):
        out = layer(x)
    # Constant down time for a given trace.
    assert torch.allclose(out[0, :, 0, :], out[0, :, 0, :1])
    # Different traces differ after conditioning.
    assert not torch.allclose(out[0, :, 0, 0], out[0, :, 3, 0])


def test_geonorm_pools_when_trace_axis_shrinks():
    layer = GeoNorm(8, geom_dim=4, groups=4)
    x = torch.randn(2, 8, 4, 16)
    emb = torch.randn(2, 16, 4)
    with geom_embedding_context(emb):
        out = layer(x)
    assert out.shape == x.shape


def test_replace_norms_swaps_batchnorm():
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    n = replace_norms_with_geonorm(net, geom_dim=4, groups=8)
    assert n == 1
    assert isinstance(net[1], GeoNorm)


def test_group_count_divides_channels():
    assert group_count(64, 8) == 8
    assert 40 % group_count(40, 8) == 0
    assert 7 % group_count(7, 8) == 0
    assert group_count(7, 8) == 7


def test_raw_geom_from_shot_receiver_coords():
    gather = {
        "trace_count": 3,
        "rec_coords": np.array([[10.0, 0.0, 5.0], [20.0, 0.0, 7.0], [30.0, 0.0, 9.0]]),
        "shot_coords": np.array([0.0, 0.0, 1.0]),
    }
    geom = raw_geom_features(gather)
    np.testing.assert_allclose(geom[:, 0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(geom[:, 1], [4.0, 6.0, 8.0])


def test_raw_geom_falls_back_to_offset_when_shot_is_origin():
    gather = {
        "trace_count": 2,
        "rec_coords": np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
        "shot_coords": np.zeros(3),
        "offset_distances": np.array([[100.0, 0.0, 0.0], [250.0, 0.0, 0.0]]),
    }
    geom = raw_geom_features(gather)
    np.testing.assert_allclose(geom[:, 0], [100.0, 250.0])
    np.testing.assert_allclose(geom[:, 1], [0.0, 0.0])


def test_normalize_geom_train_minmax():
    stats = GeomStats(dx_min=10.0, dx_max=30.0, dz_min=-4.0, dz_max=4.0)
    geom = np.array([[10.0, -4.0], [30.0, 4.0], [20.0, 0.0]], dtype=np.float32)
    out = normalize_geom(geom, stats)
    np.testing.assert_allclose(out[:, 0], [0.0, 1.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(out[:, 1], [0.0, 1.0, 0.5], atol=1e-6)


def test_attach_geom_features_writes_field():
    gather = {
        "trace_count": 2,
        "rec_coords": np.array([[0.0, 0.0, 2.0], [4.0, 0.0, 6.0]]),
        "shot_coords": np.array([0.0, 0.0, 0.0]),
        "offset_distances": np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    }
    stats = GeomStats(dx_min=1.0, dx_max=3.0, dz_min=0.0, dz_max=1.0)
    attach_geom_features(gather, stats)
    assert gather["geom_features"].shape == (2, 2)
    np.testing.assert_allclose(gather["geom_features"][:, 0], [0.0, 1.0], atol=1e-6)


@pytest.mark.parametrize(
    "raw, letter, inp, gn",
    [
        (None, "A", False, False),
        ("B", "B", True, False),
        ("C", "C", False, True),
        ("D", "D", True, True),
        ({"ablation": "C", "encoder_dim": 64}, "C", False, True),
        ({"ablation": "A", "input_channels": True, "enabled": True}, "D", True, True),
    ],
)
def test_resolve_geonorm_config(raw, letter, inp, gn):
    cfg = resolve_geonorm_config({"geonorm": raw})
    assert cfg["ablation"] == letter
    assert cfg["use_geom_input_channels"] is inp
    assert cfg["use_geonorm"] is gn


def test_cli_geonorm_overrides_recipe():
    cfg = resolve_geonorm_config({"geonorm": "B"}, cli="D")
    assert cfg["ablation"] == "D"
    assert cfg["use_geonorm"] is True


def test_fbpunet_geonorm_forward_cpu():
    from models.fbp.unet import FBPUNet
    from models.fbp.utils import prepare_input_features

    hp = {
        "unet_encoder_type": "vanilla",
        "unet_decoder_type": "vanilla",
        "encoder_block_count": 3,
        "encoder_block_channels": [16, 32, 64],
        "mid_block_channels": 64,
        "decoder_block_channels": "[64, 32, 16]",
        "decoder_attention_type": None,
        "segm_class_count": 2,
        "head_class_count": 2,
        "use_dist_offsets": False,
        "use_first_break_prior": False,
        "use_geonorm": True,
        "use_geom_input_channels": True,
        "geom_encoder_dim": 16,
        "geom_norm_groups": 8,
        "coordconv": False,
        "optimizer_type": "Adam",
        "optimizer_params": {"lr": 1e-3},
        "scheduler_type": "StepLR",
        "scheduler_params": {"step_size": 10, "gamma": 0.1},
        "update_scheduler_at_epochs": True,
        "loss_type": "crossentropy",
        "loss_params": {},
        "use_full_metrics_during_training": False,
        "eval_type": "FBPEvaluator",
        "segm_first_break_prob_threshold": 0.0,
        "eval_metrics": [{"metric_type": "MeanAbsoluteError"}],
        "gathers_to_display": 0,
        "use_checkpointing": False,
        "max_epochs": 1,
        "use_skip_connections": True,
    }
    model = FBPUNet(hp)
    model.eval()
    batch = {
        "samples": torch.randn(2, 16, 32),
        "geom_features": torch.rand(2, 16, 2),
    }
    x = prepare_input_features(
        batch, use_dist_offsets=False, use_first_break_prior=False, use_geom_input_channels=True
    )
    assert x.shape[1] == 3
    with torch.no_grad():
        logits = model(x, geom=batch["geom_features"])
    assert logits.shape[0] == 2
    assert logits.shape[1] == 2


def test_geom_encoder_and_forward_shapes():
    enc = GeomEncoder(2, 16)
    geom = torch.rand(3, 5, 2)
    emb = enc(geom)
    assert emb.shape == (3, 5, 16)
    layer = GeoNorm(8, 16, groups=8)
    x = torch.randn(3, 8, 5, 7)
    with geom_embedding_context(emb):
        out = layer(x)
    assert out.shape == x.shape
