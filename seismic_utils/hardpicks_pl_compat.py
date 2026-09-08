"""Make hardpicks BaseModel importable/runnable on PyTorch Lightning 2.x.

hardpicks was written for PL 1.5–1.9 (`EPOCH_OUTPUT`, `training_epoch_end(outputs)`, …).
Lightning AI cloud images ship PL 2.x where those APIs are gone.

Call :func:`ensure_hardpicks_lightning_compat` **before** importing
``hardpicks.models.fbp.unet`` (or any module that pulls ``hardpicks.models.base``).
"""

from __future__ import annotations

import typing

_APPLIED = False


def ensure_hardpicks_lightning_compat() -> str:
    """Stub removed PL types and rewrite epoch-end hooks for PL 2.x.

    Returns a short status string for logging.
    """
    global _APPLIED

    import pytorch_lightning as pl
    import pytorch_lightning.utilities.types as pl_types

    # Import-time annotations in hardpicks.models.base reference these aliases.
    if not hasattr(pl_types, "EPOCH_OUTPUT"):
        pl_types.EPOCH_OUTPUT = typing.Any  # type: ignore[attr-defined]
    if not hasattr(pl_types, "STEP_OUTPUT"):
        pl_types.STEP_OUTPUT = typing.Any  # type: ignore[attr-defined]

    major = int(str(pl.__version__).split(".", 1)[0])
    if major < 2:
        _APPLIED = True
        return f"pl-{pl.__version__}-noop"

    import hardpicks.models.base as base_mod

    BaseModel = base_mod.BaseModel

    def _strip_removed_hooks(cls) -> list[str]:
        removed = []
        for name in (
            "training_epoch_end",
            "validation_epoch_end",
            "test_epoch_end",
            "on_epoch_start",
            "on_epoch_end",
        ):
            if name in cls.__dict__:
                delattr(cls, name)
                removed.append(name)
        return removed

    # Always strip removed hooks — PL2 errors if they exist at all.
    stripped = _strip_removed_hooks(BaseModel)

    if getattr(BaseModel, "_seismic_pl2_compat", False):
        _APPLIED = True
        extra = f"+stripped:{','.join(stripped)}" if stripped else ""
        return f"pl-{pl.__version__}-class-already-patched{extra}"

    if _APPLIED and not stripped:
        return "already-applied"

    _orig_init = BaseModel.__init__
    _orig_train_step = BaseModel.training_step
    _orig_val_step = BaseModel.validation_step
    _orig_test_step = BaseModel.test_step
    _orig_train_epoch_start = BaseModel.on_train_epoch_start

    def _on_gpu(self) -> bool:
        try:
            return str(getattr(self, "device", "")).startswith("cuda")
        except Exception:
            return False

    def __init__(self, hyper_params):
        _orig_init(self, hyper_params)
        self._epoch_losses = {"train": [], "valid": [], "test": []}

    def training_step(self, batch, batch_idx):
        out = _orig_train_step(self, batch, batch_idx)
        loss = out["loss"] if isinstance(out, dict) else out
        self._epoch_losses.setdefault("train", []).append(loss.detach())
        return out

    def validation_step(self, batch, batch_idx):
        out = _orig_val_step(self, batch, batch_idx)
        loss = out["loss"] if isinstance(out, dict) else out
        self._epoch_losses.setdefault("valid", []).append(loss.detach())
        return out

    def test_step(self, batch, batch_idx):
        out = _orig_test_step(self, batch, batch_idx)
        loss = out["loss"] if isinstance(out, dict) else out
        self._epoch_losses.setdefault("test", []).append(loss.detach())
        return out

    def on_train_epoch_start(self):
        if _on_gpu(self):
            import torch

            torch.cuda.reset_peak_memory_stats(device=self.device)
        return _orig_train_epoch_start(self)

    def on_train_epoch_end(self):
        losses = self._epoch_losses.get("train", [])
        self._epoch_losses["train"] = []
        self._generic_epoch_end(
            prefix="train", losses=losses, evaluator=self.train_evaluator
        )
        if self.update_scheduler_at_epochs:
            self.scheduler.step()
        if _on_gpu(self):
            import torch

            max_mem_mb = torch.cuda.max_memory_allocated(device=self.device) // (
                1024 * 1024
            )
            self.log("maximum_cuda_memory_mb", float(max_mem_mb))
            base_mod.sigopt.log_metric("maximum_cuda_memory_mb", max_mem_mb)
        base_mod.sigopt.log_metric("completed_epochs", self.current_epoch + 1)

    def on_validation_epoch_end(self):
        losses = self._epoch_losses.get("valid", [])
        self._epoch_losses["valid"] = []
        self._generic_epoch_end(
            prefix="valid", losses=losses, evaluator=self.valid_evaluator
        )

    def on_test_epoch_end(self):
        losses = self._epoch_losses.get("test", [])
        self._epoch_losses["test"] = []
        self._generic_epoch_end(
            prefix="test", losses=losses, evaluator=self.test_evaluator
        )

    BaseModel.__init__ = __init__
    BaseModel.training_step = training_step
    BaseModel.validation_step = validation_step
    BaseModel.test_step = test_step
    BaseModel.on_train_epoch_start = on_train_epoch_start
    BaseModel.on_train_epoch_end = on_train_epoch_end
    BaseModel.on_validation_epoch_end = on_validation_epoch_end
    BaseModel.on_test_epoch_end = on_test_epoch_end
    _strip_removed_hooks(BaseModel)

    BaseModel._seismic_pl2_compat = True
    _APPLIED = True
    return f"pl-{pl.__version__}-patched"
