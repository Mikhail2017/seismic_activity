"""Regression coverage for the model reload used by post-fit validation."""

import builtins
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pytest
import pytorch_lightning as pl
import torch
from mlflow.exceptions import MlflowException

from seismic_utils.hardpicks_pl_compat import ensure_hardpicks_lightning_compat
from seismic_utils.pickers import attach_smooth_evaluators
from train.fbp_train import (
    _load_fbpunet_from_checkpoint,
    build_model_config,
    make_model_checkpoint,
    parse_args,
    validate_best_checkpoint,
)


@pytest.fixture
def checkpoint_factory(tmp_path, monkeypatch):
    ensure_hardpicks_lightning_compat()
    from models.fbp.unet import FBPUNet

    # Keep tests offline and avoid creating an MLflow run/database.
    monkeypatch.setattr(mlflow, "log_param", Mock())

    def create(picker):
        before_after = picker != "fbpunet"
        hp, _ = build_model_config(
            model="vanilla-before-after" if before_after else "vanilla",
            max_epochs=1,
            smooth_threshold=7,
        )
        hp.update(
            encoder_block_count=2,
            encoder_block_channels=[4, 8],
            mid_block_channels=8,
            decoder_block_channels=[8, 4],
            use_dist_offsets=False,
        )
        if picker is None:
            hp.pop("picker")  # Older before/after checkpoints only stored class count.
        model = FBPUNet(hp)
        if before_after:
            attach_smooth_evaluators(model, hp)
        path = tmp_path / "best.ckpt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "hparams_name": "hyper_params",
                "pytorch-lightning_version": pl.__version__,
                "epoch": 0,
                "global_step": 1,
            },
            path,
        )
        return model, path

    return create


def _perfect_predictions(before_after):
    labels = torch.tensor([[20, 40, 60, 80]])
    samples = torch.arange(128).view(1, 1, -1)
    positive = samples >= labels.unsqueeze(-1) if before_after else samples == labels.unsqueeze(-1)
    logits = torch.stack((~positive, positive), dim=1).float() * 12 - 6
    batch = {
        "batch_size": 1,
        "samples": torch.zeros(1, 4, 128),
        "rec_ids": torch.tensor([[1, 2, 3, 4]]),
        "offset_distances": torch.ones(1, 4, 3),
        "origin": ["synthetic"],
        "shot_id": torch.tensor([1]),
        "gather_id": torch.tensor([1]),
        "first_break_labels": labels,
        "segmentation_mask": positive.long(),
    }
    return batch, logits


@pytest.mark.parametrize("picker", ["before_after", None, "fbpunet"])
@pytest.mark.parametrize("mlflow_conflict", [False, True])
def test_checkpoint_restores_evaluators(checkpoint_factory, monkeypatch, picker, mlflow_conflict):
    from hardpicks.metrics.base import NoneEvaluator
    from hardpicks.metrics.fbp.evaluator import FBPEvaluator

    model, path = checkpoint_factory(picker)
    log_param = Mock(side_effect=MlflowException("parameter already logged") if mlflow_conflict else None)
    monkeypatch.setattr(mlflow, "log_param", log_param)
    loaded = _load_fbpunet_from_checkpoint(path)
    assert mlflow.log_param is log_param
    assert log_param.called

    before_after = picker != "fbpunet"
    for name in ("valid_evaluator", "test_evaluator", "pred_evaluator"):
        evaluator = getattr(loaded, name)
        assert type(evaluator) is type(getattr(model, name))
        if before_after:
            assert evaluator.smooth_threshold == 7
        else:
            assert type(evaluator) is FBPEvaluator
    assert isinstance(loaded.train_evaluator, NoneEvaluator)

    # The reload must preserve weights, forward outputs, and loss as well.
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    model.eval()
    loaded.eval()
    inputs = torch.randn(1, 1, 4, 128)
    with torch.no_grad():
        torch.testing.assert_close(loaded(inputs), model(inputs), rtol=0, atol=0)

    batch, logits = _perfect_predictions(before_after)
    expected = model.valid_evaluator.ingest(batch, 0, logits)
    actual = loaded.valid_evaluator.ingest(batch, 0, logits)
    assert actual == pytest.approx(expected)
    assert actual["GatherCoverage"] == 1.0
    assert actual["HitRate1px"] == 1.0
    for metric in ("MeanAbsoluteError", "MeanBiasError", "RootMeanSquaredError"):
        assert actual[metric] == 0.0
    torch.testing.assert_close(
        loaded.loss_fn(logits, batch["segmentation_mask"]),
        model.loss_fn(logits, batch["segmentation_mask"]),
    )


def test_checkpoint_restores_evaluators_without_mlflow(checkpoint_factory, monkeypatch):
    model, path = checkpoint_factory("before_after")
    original_import = builtins.__import__

    def import_without_mlflow(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("MLflow unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_mlflow)
    loaded = _load_fbpunet_from_checkpoint(path)
    assert type(loaded.valid_evaluator) is type(model.valid_evaluator)
    assert loaded.valid_evaluator.smooth_threshold == 7


def test_final_lightning_validation_matches_before_reload(checkpoint_factory, monkeypatch):
    model, path = checkpoint_factory("before_after")
    loaded = _load_fbpunet_from_checkpoint(path)
    batch, logits = _perfect_predictions(before_after=True)
    # Exercise real Lightning validation with deterministic, perfect predictions.
    monkeypatch.setattr(type(model), "forward", lambda self, inputs: logits.to(inputs.device))
    loader = torch.utils.data.DataLoader([batch], batch_size=None)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    expected = trainer.validate(model, dataloaders=loader, verbose=False)[0]
    actual = trainer.validate(loaded, dataloaders=loader, verbose=False)[0]
    assert actual == pytest.approx(expected)
    assert actual["valid/GatherCoverage"] == 1.0
    assert actual["valid/HitRate1px"] == 1.0
    assert actual["valid/MeanAbsoluteError"] == 0.0
    assert actual["valid/MeanBiasError"] == 0.0
    assert actual["valid/RootMeanSquaredError"] == 0.0
    assert actual["valid/loss"] < 0.001


def test_checkpoint_load_failure_restores_mlflow(checkpoint_factory, monkeypatch):
    _, path = checkpoint_factory("before_after")
    from models.fbp.unet import FBPUNet

    original_log_param = mlflow.log_param
    monkeypatch.setattr(FBPUNet, "load_from_checkpoint", Mock(side_effect=RuntimeError("bad checkpoint")))
    with pytest.raises(RuntimeError, match="bad checkpoint"):
        _load_fbpunet_from_checkpoint(path)
    assert mlflow.log_param is original_log_param


def test_make_model_checkpoint_keeps_version_counter(tmp_path):
    callback = make_model_checkpoint(tmp_path)
    assert callback.save_top_k == 1
    assert callback.monitor == "valid/HitRate1px"
    assert getattr(callback, "_enable_version_counter", True) is True


def test_make_model_checkpoint_honors_save_top_k(tmp_path):
    callback = make_model_checkpoint(tmp_path, save_top_k=-1)
    assert callback.save_top_k == -1


def test_parse_args_save_top_k_from_recipe_and_cli(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text("save_top_k: 3\n")
    assert parse_args(["--config", str(cfg)]).save_top_k == 3
    assert parse_args(["--config", str(cfg), "--save-top-k", "-1"]).save_top_k == -1
    with pytest.raises(SystemExit):
        parse_args(["--config", str(cfg), "--save-top-k", "-2"])


def test_post_fit_validate_matches_in_training_metrics(checkpoint_factory, tmp_path, monkeypatch):
    """Revalidation must preserve results without overwriting existing checkpoints."""
    monkeypatch.setattr(mlflow, "log_param", Mock())
    model, _ = checkpoint_factory("before_after")
    batch, _ = _perfect_predictions(before_after=True)
    loader = torch.utils.data.DataLoader([batch], batch_size=None)
    leftover = tmp_path / "best-epoch=000-step=000001.ckpt"
    leftover.write_bytes(b"stale-previous-run")
    checkpoint_cb = make_model_checkpoint(tmp_path)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint_cb],
        limit_train_batches=1,
        limit_val_batches=1,
    )
    trainer.fit(model, loader, loader)
    fit_metrics = trainer.callback_metrics
    assert "valid/HitRate1px" in fit_metrics
    best_path = Path(checkpoint_cb.best_model_path)
    assert best_path.is_file()
    assert "-v1" in best_path.name
    assert leftover.read_bytes() == b"stale-previous-run"
    assert best_path.stat().st_size > 64

    actual = validate_best_checkpoint(trainer, loader, best_path)[0]
    assert actual["valid/HitRate1px"] == pytest.approx(float(fit_metrics["valid/HitRate1px"]), abs=1e-5)
    assert actual["valid/loss"] == pytest.approx(float(fit_metrics["valid/loss"]), rel=1e-4, abs=1e-5)
    assert actual["valid/MeanAbsoluteError"] == pytest.approx(
        float(fit_metrics["valid/MeanAbsoluteError"]), rel=1e-4, abs=1e-5
    )