import copy

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from conftest import make_model, make_parser
from train.fbp_train import build_loaders, make_model_checkpoint, validate_best_checkpoint, _load_fbpunet_from_checkpoint


def test_repeated_reads_and_crop_preserve_metadata(tiny_hdf5):
    from hardpicks.data.fbp.gather_transforms import crop_samples, flip, drop_traces

    parser = make_parser(tiny_hdf5)
    raw = parser.dataset.dataset
    original = copy.deepcopy(raw.get_meta_gather(0))
    first = copy.deepcopy(parser[0])
    for _ in range(4):
        item = parser[0]
        for key in ("samples", "offset_distances", "first_break_labels", "segmentation_mask"):
            np.testing.assert_array_equal(first[key], item[key])
    item = raw[0]
    crop_samples(item, 64)
    flip(item)
    drop_traces(item, 1, False, True)
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(raw.get_meta_gather(0)[key], value)
    np.testing.assert_array_equal(parser[0]["first_break_labels"], [20, 40, 60, 80])
    raw._worker_h5fd.close()


def test_train_crop_augmentation_preserves_future_labels(tiny_hdf5):
    parser = make_parser(tiny_hdf5, augmentations=[
        {"type": "crop", "params": {"low_sample_count": 64, "high_sample_count": 65, "max_crop_fraction": 0.5}},
        {"type": "flip"},
    ])
    raw = parser.dataset.dataset
    expected = raw.get_meta_gather(0)["first_break_labels"].copy()
    for _ in range(4):
        item = parser[0]
        assert item["sample_count"] < 80
        assert (item["first_break_labels"] == 0).any()
        np.testing.assert_array_equal(raw.get_meta_gather(0)["first_break_labels"], expected)
    parser.augmentations = None
    np.testing.assert_array_equal(parser[0]["first_break_labels"], expected)
    raw._worker_h5fd.close()


class ValidationAudit(pl.Callback):
    def __init__(self):
        self.rows = []
        self.offsets = []

    def on_validation_epoch_start(self, trainer, model):
        self.bn = {k: v.clone() for k, v in model.state_dict().items() if "running_" in k or "num_batches_tracked" in k}

    def on_validation_batch_start(self, trainer, model, batch, batch_idx):
        assert not any(m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
        if batch_idx == 0:
            self.offsets.append(batch["offset_distances"].clone())

    def on_validation_end(self, trainer, model):
        for key, value in self.bn.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        if not trainer.sanity_checking:
            self.rows.append({k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("valid/")})


@pytest.mark.parametrize("workers", [0, 2])
def test_multi_epoch_hdf5_checkpoint_matches_fresh_eval(tiny_hdf5, tmp_path, workers):
    pl.seed_everything(0)
    train, valid = make_parser(tiny_hdf5), make_parser(tiny_hdf5)
    tl, vl = build_loaders(train, valid, 2, workers)
    model = make_model()
    audit = ValidationAudit()
    checkpoint = make_model_checkpoint(tmp_path / "checkpoints", save_top_k=-1)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=3, limit_train_batches=2,
                         logger=False, enable_progress_bar=False, enable_model_summary=False,
                         callbacks=[audit, checkpoint])
    trainer.fit(model, tl, vl)
    path = sorted((tmp_path / "checkpoints").glob("best*.ckpt"))[-1]
    expected = dict(audit.rows[-1])
    actual = validate_best_checkpoint(trainer, vl, path)[0]
    assert actual == pytest.approx(expected, abs=1e-7)
    loaded = _load_fbpunet_from_checkpoint(path)
    actual = trainer.validate(loaded, dataloaders=vl, verbose=False)[0]
    assert actual == pytest.approx(expected, abs=1e-7)
    fresh = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, enable_model_summary=False)
    actual = fresh.validate(loaded, dataloaders=vl, verbose=False)[0]
    assert actual == pytest.approx(expected, abs=1e-7)
    for offsets in audit.offsets:
        torch.testing.assert_close(offsets, audit.offsets[0], rtol=0, atol=0)
    # Exercise the standalone evaluator too, including a fresh parser/worker pool.
    from train.fbp_eval import run_eval
    from seismic_utils.pickers import make_eval_evaluator

    _, fresh_loader = build_loaders(make_parser(tiny_hdf5), make_parser(tiny_hdf5), 2, 0)
    _, metrics, loss = run_eval(loaded, fresh_loader, torch.device("cpu"),
                               make_eval_evaluator(loaded.hparams, "before_after"))
    assert {f"valid/{k}": v for k, v in metrics.items()} == pytest.approx(
        {k: v for k, v in expected.items() if k != "valid/loss"}, rel=1e-6, abs=1e-7)
    assert loss == pytest.approx(expected["valid/loss"], abs=1e-7)