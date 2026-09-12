import copy
import json
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from conftest import make_model, make_parser
from seismic_utils.predict import resolve_checkpoint
from seismic_utils.training_state import load_checkpoint, reserve_run_directory, validate_resume_checkpoint
from train.fbp_train import build_loaders, build_model_config, _find_metrics_csv, make_model_checkpoint


class LearningRates(pl.Callback):
    def __init__(self):
        self.values = []

    def on_train_epoch_start(self, trainer, model):
        self.values.append(trainer.optimizers[0].param_groups[0]["lr"])


def test_scheduler_resume_matches_uninterrupted(tiny_hdf5, tmp_path):
    def fit(epochs, ckpt=None):
        model = make_model(max_epochs=4)
        model.hparams.scheduler_params["step_size"] = 2
        tl, vl = build_loaders(make_parser(tiny_hdf5), make_parser(tiny_hdf5), 2, 0)
        rates = LearningRates()
        trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=epochs, logger=False,
                             enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                             limit_train_batches=1, limit_val_batches=1, callbacks=[rates])
        trainer.fit(model, tl, vl, ckpt_path=str(ckpt) if ckpt else None)
        return trainer, rates.values

    _, continuous = fit(4)
    trainer, first = fit(1)
    path = tmp_path / "resume.ckpt"
    trainer.save_checkpoint(path)
    assert load_checkpoint(path)["seismic_scheduler_state"]["last_epoch"] == 1
    _, rest = fit(4, path)
    assert first + rest == pytest.approx(continuous)


@pytest.mark.parametrize("key,value", [
    ("picker", "fbpunet"), ("use_dist_offsets", False), ("loss_type", "dice"),
    ("segm_first_break_smooth_threshold", 99), ("training_data", {"backend": "npz"}),
])
def test_resume_rejects_semantic_changes(key, value):
    hp, _ = build_model_config(model="resnet34-before-after", max_epochs=4)
    ckpt = {"hyper_parameters": copy.deepcopy(hp), "seismic_scheduler_state": {}}
    hp[key] = value
    with pytest.raises(ValueError, match="[Rr]esume"):
        validate_resume_checkpoint(ckpt, hp)


def test_legacy_resume_rejected_but_weights_still_load():
    hp, _ = build_model_config(model="resnet34-before-after", max_epochs=4)
    with pytest.raises(ValueError, match="scheduler state"):
        validate_resume_checkpoint({"hyper_parameters": hp}, hp)


def test_artifacts_are_unambiguous(tmp_path):
    root = tmp_path / "run"
    reserve_run_directory(root)
    with pytest.raises(ValueError, match="not empty"):
        reserve_run_directory(root)
    a, b = root / "best-a.ckpt", root / "best-b.ckpt"
    a.touch()
    assert resolve_checkpoint(ckpt_dir=root) == a
    b.touch()
    with pytest.raises(FileNotFoundError, match="Multiple"):
        resolve_checkpoint(ckpt_dir=root)
    (root / "best_checkpoint.json").write_text(json.dumps({"path": a.name}))
    assert resolve_checkpoint(ckpt_dir=root) == a
    for version in (9, 10):
        path = root / "logs" / f"version_{version}" / "metrics.csv"
        path.parent.mkdir(parents=True)
        path.touch()
    assert _find_metrics_csv(root / "logs").parent.name == "version_10"


def test_model_config_smoothing_is_preserved(tmp_path):
    config = tmp_path / "model.yaml"
    config.write_text("segm_first_break_smooth_threshold: 7\n")
    hp, _ = build_model_config(model="vanilla-before-after", max_epochs=1, model_config_path=config)
    assert hp["segm_first_break_smooth_threshold"] == 7
    hp, _ = build_model_config(model="vanilla-before-after", max_epochs=1, model_config_path=config, smooth_threshold=9)
    assert hp["segm_first_break_smooth_threshold"] == 9


def test_training_entrypoint_and_report(tiny_hdf5, tmp_path, monkeypatch):
    import train.fbp_train as cli

    monkeypatch.delenv("SEISMIC_RUN_ID", raising=False)
    monkeypatch.setattr(cli, "resolve_hardpicks_site_info", lambda name, data_dir: {**tiny_hdf5, "site_name": name})
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("backend: hdf5\nmodel: vanilla-before-after\nepochs: 3\nbatch_size: 2\nnum_workers: 2\npatience: 0\nsave_top_k: -1\naugmentations: []\n")
    model_config = tmp_path / "model.yaml"
    model_config.write_text("encoder_block_count: 2\nencoder_block_channels: [4, 8]\nmid_block_channels: 8\ndecoder_block_channels: [8, 4]\nsegm_first_break_smooth_threshold: 7\n")
    output = tmp_path / "run"
    args = ["--config", str(recipe), "--model-config", str(model_config), "--fold", "A",
            "--output-dir", str(output), "--report-dir", str(tmp_path / "reports"), "--accelerator", "cpu"]
    assert cli.main(args) == 0
    report = json.loads(next((tmp_path / "reports").rglob("final.json")).read_text())
    checkpoint = load_checkpoint(resolve_checkpoint(ckpt_dir=output))
    assert report["final_validation"] == pytest.approx(checkpoint["seismic_validation_metrics"], rel=1e-5, abs=1e-6)
    assert report["best_score"] == pytest.approx(report["final_validation"]["valid/HitRate1px"])
    assert checkpoint["hyper_parameters"]["training_data"]["preprocessing_version"]
    assert (output / "epoch_metrics.csv").is_file()
    # A real CLI resume into a new run retains the same task/preprocessing.
    resume_args = args.copy()
    resume_args[resume_args.index(str(output))] = str(tmp_path / "resumed")
    monkeypatch.delenv("SEISMIC_RUN_ID", raising=False)
    assert cli.main(resume_args + ["--epochs", "4", "--ckpt", str(resolve_checkpoint(ckpt_dir=output))]) == 0