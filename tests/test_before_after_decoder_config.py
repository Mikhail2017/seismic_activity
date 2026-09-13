"""Decoder CLI/config precedence, checkpoint metadata, and report identity."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat

ensure_hardpicks_lightning_compat()

from train import fbp_eval, fbp_train
from seismic_utils.fbp_eval_report import write_report_md
from seismic_utils.training_state import validate_resume_checkpoint


def test_training_config_precedence_and_resume(tmp_path):
    config_path = tmp_path / "model.yaml"
    config_path.write_text("before_after_decoder: legacy\n")
    kwargs = dict(model="resnet18", max_epochs=2, picker="before_after")
    baseline, _ = fbp_train.build_model_config(**kwargs)
    assert baseline["before_after_decoder"] == "legacy"
    recipe = {"before_after_decoder": "change_point"}
    recipe_config, _ = fbp_train.build_model_config(**kwargs, recipe=recipe)
    assert recipe_config["before_after_decoder"] == "change_point"
    file_config, _ = fbp_train.build_model_config(**kwargs, recipe=recipe, model_config_path=config_path)
    assert file_config["before_after_decoder"] == "legacy"
    cli_config, _ = fbp_train.build_model_config(
        **kwargs, recipe=recipe, model_config_path=config_path, before_after_decoder="change_point"
    )
    assert cli_config["before_after_decoder"] == "change_point"
    validate_resume_checkpoint({"hyper_parameters": cli_config, "seismic_scheduler_state": {}}, recipe_config)
    with pytest.raises(ValueError, match="before_after_decoder"):
        validate_resume_checkpoint({"hyper_parameters": cli_config, "seismic_scheduler_state": {}}, baseline)


def test_invalid_decoder_and_wrong_task():
    with pytest.raises(ValueError, match="Unknown"):
        fbp_train.build_model_config(
            model="resnet18", max_epochs=1, picker="before_after", before_after_decoder="typo"
        )
    with pytest.raises(ValueError, match="requires"):
        fbp_train.build_model_config(model="resnet18", max_epochs=1, before_after_decoder="change_point")


def test_cli_accepts_decoder_and_preserves_unspecified_default(tmp_path):
    config = tmp_path / "recipe.yaml"
    config.write_text("{}\n")
    train_args = fbp_train.parse_args([
        "--config", str(config), "--fold", "A", "--picker", "before_after",
        "--before-after-decoder", "change_point",
    ])
    assert train_args.before_after_decoder == "change_point"
    assert fbp_train.parse_args(["--config", str(config), "--fold", "A"]).before_after_decoder is None
    eval_base = ["--ckpt", str(tmp_path / "weights.ckpt"), "--fold", "A"]
    assert fbp_eval.parse_args(eval_base).before_after_decoder is None
    assert fbp_eval.parse_args(eval_base + ["--before-after-decoder", "change_point"]).before_after_decoder == "change_point"
    with pytest.raises(SystemExit):
        fbp_eval.parse_args(eval_base + ["--before-after-decoder", "typo"])


@pytest.mark.parametrize("decoder", ["legacy", "change_point"])
def test_report_records_decoder(tmp_path, decoder):
    report = write_report_md(
        tmp_path / "report.md", meta={"picker": "before_after"},
        metrics={"before_after_decoder": decoder}, offset_df=pd.DataFrame(),
        worst_names=[], typical_names=[],
    )
    assert f"**Before/after decoder:** {decoder}" in report.read_text()


@pytest.mark.parametrize("override,expected", [(None, 10), ("change_point", 200)])
def test_eval_cli_runs_decoder_and_writes_metrics(tmp_path, monkeypatch, override, expected):
    """Exercise main/report generation with deterministic logits, no survey data."""
    hp = {
        "picker": "before_after", "segm_class_count": 2,
        "segm_first_break_prob_threshold": 0., "unet_encoder_type": "test",
        "training_data": {"batch_size": 1},
    }
    logits = torch.zeros((1, 2, 2, 512))
    logits[:, 0, :, :200] = 5
    logits[:, 1, :, 200:] = 5
    logits[:, 0, :, [10, 80]] = 0
    logits[:, 1, :, [10, 80]] = 5
    class Model:
        hparams = hp
        segm_class_count = 2

        def to(self, device):
            return self

        def eval(self):
            return self

        def _generic_step(self, batch, batch_idx, evaluator):
            self._last_eval_loss_weight = 1
            return logits, torch.tensor(0.), evaluator.ingest(batch, batch_idx, logits)

    class Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            target = np.zeros((2, 400), dtype=np.int64)
            target[:, 200:] = 1
            return {
                "samples": np.ones((2, 400), dtype=np.float32),
                "segmentation_mask": target, "first_break_labels": np.array([200, 200]),
                "trace_count": 2, "sample_count": 400, "sample_rate_ms": 2.,
                "rec_ids": np.array([1, 2]), "offset_distances": np.zeros((2, 3), dtype=np.float32),
                "origin": "Halfmile", "shot_id": 1, "gather_id": 1,
            }

        def get_meta_gather(self, index):
            return self[index]

    ckpt = tmp_path / "weights.ckpt"
    ckpt.touch()
    monkeypatch.setattr(fbp_eval, "load_fbp_model", lambda path: Model())
    monkeypatch.setattr(fbp_eval.train_cli, "build_split_parser", lambda *a, **k: Dataset())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = [
        "--ckpt", str(ckpt), "--fold", "A", "--num-workers", "0",
        "--report-dir", str(tmp_path / "reports"), "--n-worst", "0", "--n-typical", "0",
    ]
    if override:
        args += ["--before-after-decoder", override, "--smooth-threshold", "999"]
    assert fbp_eval.main(args) == 0
    metrics_path = next((tmp_path / "reports").glob("*/metrics.json"))
    metrics = json.loads(metrics_path.read_text())
    assert metrics["before_after_decoder"] == (override or "legacy")
    assert metrics["smooth_threshold"] == (None if override else 50)
    assert metrics["MeanAbsoluteError"] == abs(expected - 200)
    assert metrics["evaluator"]["MeanAbsoluteError"] == abs(expected - 200)
    assert metrics["GatherCoverage"] == 1.
    assert "before_after_decoder" not in hp  # the checkpoint metadata is untouched
    assert f"**Before/after decoder:** {override or 'legacy'}" in metrics_path.with_name("report.md").read_text()


@pytest.mark.parametrize("saved_decoder", [None, "change_point"])
def test_real_model_optimizer_step_and_fresh_checkpoint_parity(tmp_path, monkeypatch, saved_decoder):
    """A decoder setting changes neither weights nor checkpoint loading semantics."""
    import pytorch_lightning as pl
    import hardpicks.utils.hp_utils as hp_utils
    from models.fbp.unet import FBPUNet
    from seismic_utils.pickers import attach_smooth_evaluators, decode_nn_picks
    from seismic_utils.predict import load_fbp_model

    # Keep external experiment logging out of an isolated unit test.
    monkeypatch.setattr(hp_utils, "log_hp", lambda *a, **k: None)
    config, _ = fbp_train.build_model_config(model="vanilla", max_epochs=1, picker="before_after")
    config.update({
        "encoder_block_count": 2, "encoder_block_channels": [4, 8],
        "mid_block_channels": 16, "decoder_block_channels": "[8, 4]",
        "use_dist_offsets": False, "segm_first_break_smooth_threshold": 3,
    })
    if saved_decoder is None:
        config.pop("before_after_decoder")
    else:
        config["before_after_decoder"] = saved_decoder
    model = FBPUNet(config)
    attach_smooth_evaluators(model, config)
    x = torch.randn((1, 1, 8, 32), generator=torch.Generator().manual_seed(17))
    target = torch.zeros((1, 8, 32), dtype=torch.long)
    target[:, :, 16:] = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.functional.cross_entropy(model(x), target)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected_logits = model(x)
        expected = decode_nn_picks(expected_logits, model, sample_counts=[29])
    ckpt = tmp_path / "model.ckpt"
    torch.save({
        "state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
        "hparams_name": "hyper_params", "pytorch-lightning_version": pl.__version__,
    }, ckpt)
    restored = fbp_train._load_fbpunet_from_checkpoint(ckpt).eval()
    assert restored.valid_evaluator.before_after_decoder == (saved_decoder or "legacy")
    inference = load_fbp_model(ckpt, device="cpu")
    for loaded in (restored, inference):
        with torch.no_grad():
            actual_logits = loaded(x)
        torch.testing.assert_close(actual_logits, expected_logits)
        actual = decode_nn_picks(actual_logits, loaded, sample_counts=[29])
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, equal_nan=True)