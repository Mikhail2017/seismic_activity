from types import SimpleNamespace

import pytest

from train.fbp_train import validate_best_checkpoint


def test_revalidation_checks_all_metrics_not_just_hits(tmp_path):
    module = SimpleNamespace(_checkpoint_validation_metrics={"valid/HitRate1px": 0.7, "valid/RootMeanSquaredError": 7.5})
    trainer = SimpleNamespace(lightning_module=module, validate=lambda *a, **k: [{"valid/HitRate1px": 0.7, "valid/RootMeanSquaredError": 32.0}])
    with pytest.raises(RuntimeError, match="RootMeanSquaredError"):
        validate_best_checkpoint(trainer, [], tmp_path / "best.ckpt")