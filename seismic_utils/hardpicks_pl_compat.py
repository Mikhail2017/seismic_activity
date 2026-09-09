"""Make hardpicks BaseModel importable/runnable on PyTorch Lightning 2.x.

hardpicks was written for PL 1.5–1.9 (`EPOCH_OUTPUT`, `training_epoch_end(outputs)`, …).
Lightning AI cloud images ship PL 2.x where those APIs are gone.

Call :func:`ensure_hardpicks_lightning_compat` **before** importing
``hardpicks.models.fbp.unet`` (or any module that pulls ``hardpicks.models.base``).
"""

from __future__ import annotations

import typing

_APPLIED = False
_COMPAT_VERSION = 2  # bump when patch behavior changes


def _patch_numpy_nan_alias() -> None:
    """hardpicks uses ``np.NaN``, removed in NumPy 2."""
    import numpy as np

    if not hasattr(np, "NaN"):
        np.NaN = np.nan  # type: ignore[attr-defined]


def ensure_hardpicks_lightning_compat() -> str:
    """Stub removed PL types and rewrite epoch-end hooks for PL 2.x.

    Returns a short status string for logging.
    """
    global _APPLIED

    _patch_numpy_nan_alias()

    import pytorch_lightning as pl
    import pytorch_lightning.utilities.types as pl_types
    import torch.utils.data

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

    def _unwrap_loader(obj):
        """Normalize CombinedLoader / list / dict / DataLoader from PL1 or PL2."""
        seen = set()
        while id(obj) not in seen:
            seen.add(id(obj))
            if isinstance(obj, torch.utils.data.DataLoader):
                return obj
            if obj is None:
                return None
            # CombinedLoader (PL2): .loaders may be DataLoader, list, dict, …
            loaders = getattr(obj, "loaders", None)
            if loaders is not None and loaders is not obj:
                obj = loaders
                continue
            if isinstance(obj, dict):
                if len(obj) == 1:
                    obj = next(iter(obj.values()))
                    continue
                raise AssertionError(
                    f"expected a single dataloader, got dict keys={list(obj.keys())}"
                )
            if isinstance(obj, (list, tuple)):
                if len(obj) == 1:
                    obj = obj[0]
                    continue
                raise AssertionError(
                    f"expected a single dataloader, got sequence len={len(obj)}"
                )
            break
        if not isinstance(obj, torch.utils.data.DataLoader):
            raise AssertionError(
                f"could not unwrap trainer dataloader to DataLoader, got {type(obj)}"
            )
        return obj

    def _get_dataloader_from_trainer(trainer, dataloader_name):
        """PL1/PL2-compatible replacement for BaseModel._get_dataloader_from_trainer."""
        # Map hardpicks names → attribute candidates on Trainer.
        aliases = {
            "train_dataloader": (
                "train_dataloader",
                "train_dataloaders",
            ),
            "val_dataloader": (
                "val_dataloader",
                "val_dataloaders",
            ),
            "test_dataloader": (
                "test_dataloader",
                "test_dataloaders",
            ),
        }
        names = aliases.get(dataloader_name, (dataloader_name,))
        last_exc: Exception | None = None
        for name in names:
            if not hasattr(trainer, name):
                continue
            try:
                return _unwrap_loader(getattr(trainer, name))
            except Exception as exc:  # noqa: BLE001 — try next alias
                last_exc = exc

        # PL2 fallbacks via loop objects when Trainer attrs are absent/empty.
        loop_paths = {
            "train_dataloader": (
                ("fit_loop", "_combined_loader"),
                ("fit_loop", "epoch_loop", "_data_fetcher", "_dataset"),
            ),
            "val_dataloader": (
                ("fit_loop", "epoch_loop", "val_loop", "_combined_loader"),
                ("validate_loop", "_combined_loader"),
            ),
            "test_dataloader": (
                ("test_loop", "_combined_loader"),
            ),
        }
        for path in loop_paths.get(dataloader_name, ()):
            obj = trainer
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                return _unwrap_loader(obj)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

        raise AttributeError(
            f"Trainer has no usable dataloader for {dataloader_name!r}"
        ) from last_exc

    # Always strip removed hooks — PL2 errors if they exist at all.
    stripped = _strip_removed_hooks(BaseModel)

    # Always install the PL2 dataloader getter (idempotent).
    BaseModel._get_dataloader_from_trainer = staticmethod(_get_dataloader_from_trainer)

    prev = getattr(BaseModel, "_seismic_pl2_compat", 0)
    # Old builds used True (==1). Treat any prior patch as "hooks already wrapped".
    if prev >= _COMPAT_VERSION:
        _APPLIED = True
        extra = f"+stripped:{','.join(stripped)}" if stripped else ""
        return f"pl-{pl.__version__}-class-already-patched-v{prev}{extra}"

    def _on_gpu(self) -> bool:
        try:
            return str(getattr(self, "device", "")).startswith("cuda")
        except Exception:
            return False

    def on_train_epoch_start(self):
        if _on_gpu(self):
            import torch

            torch.cuda.reset_peak_memory_stats(device=self.device)
        self.train_evaluator.reset()
        assert self.scheduler is not None, "need to define a scheduler before training!"
        if self.images_to_display:
            self._pick_data_ids_to_render_and_log(
                prefix="train", data_loader=self._get_train_dataloader()
            )

    def on_validation_epoch_start(self):
        self.valid_evaluator.reset()
        if self.images_to_display:
            self._pick_data_ids_to_render_and_log(
                prefix="valid", data_loader=self._get_val_dataloader()
            )

    def on_test_epoch_start(self):
        self.test_evaluator.reset()
        if self.images_to_display:
            self._pick_data_ids_to_render_and_log(
                prefix="test", data_loader=self._get_test_dataloader()
            )

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

    # Upgrade path: prior compat already wrapped steps — only refresh epoch hooks.
    if prev:
        BaseModel.on_train_epoch_start = on_train_epoch_start
        BaseModel.on_validation_epoch_start = on_validation_epoch_start
        BaseModel.on_test_epoch_start = on_test_epoch_start
        BaseModel.on_train_epoch_end = on_train_epoch_end
        BaseModel.on_validation_epoch_end = on_validation_epoch_end
        BaseModel.on_test_epoch_end = on_test_epoch_end
        _strip_removed_hooks(BaseModel)
        BaseModel._seismic_pl2_compat = _COMPAT_VERSION
        _APPLIED = True
        return f"pl-{pl.__version__}-upgraded-v{_COMPAT_VERSION}"

    _orig_init = BaseModel.__init__
    _orig_train_step = BaseModel.training_step
    _orig_val_step = BaseModel.validation_step
    _orig_test_step = BaseModel.test_step

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

    BaseModel.__init__ = __init__
    BaseModel.training_step = training_step
    BaseModel.validation_step = validation_step
    BaseModel.test_step = test_step
    BaseModel.on_train_epoch_start = on_train_epoch_start
    BaseModel.on_validation_epoch_start = on_validation_epoch_start
    BaseModel.on_test_epoch_start = on_test_epoch_start
    BaseModel.on_train_epoch_end = on_train_epoch_end
    BaseModel.on_validation_epoch_end = on_validation_epoch_end
    BaseModel.on_test_epoch_end = on_test_epoch_end
    _strip_removed_hooks(BaseModel)

    BaseModel._seismic_pl2_compat = _COMPAT_VERSION
    _APPLIED = True
    return f"pl-{pl.__version__}-patched-v{_COMPAT_VERSION}"
