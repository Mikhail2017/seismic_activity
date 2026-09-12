"""CPU/Gloo checks with uneven shards and duplicated sampler padding."""
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from seismic_utils.validation import mean_epoch_loss, summarize_distributed_evaluator, unique_validation_batch


def distributed_worker(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        sampler = DistributedSampler(range(5), num_replicas=2, rank=rank, shuffle=False)
        indices = torch.tensor(list(sampler))
        batch, keep = unique_validation_batch({"_eval_index": indices, "batch_size": len(indices),
                                              "origin": ["site"] * len(indices)})
        selected = batch["_eval_index"].numpy()
        errors = np.array([0., 1., 2., 3., 10.])[selected]
        # Include a site seen only on one rank to test category union.
        frame = pd.DataFrame({"OriginId": selected % 2, "Errors": errors,
                              "GatherCoverage": selected != 4, "ExpectedCoverage": True})
        metrics = {
            "HitRate1px": ("HitRate", {"buffer_size_px": 1}),
            "MeanAbsoluteError": ("MeanAbsoluteError", {}),
            "MeanBiasError": ("MeanBiasError", {}),
            "RootMeanSquaredError": ("RootMeanSquaredError", {}),
            "GatherCoverage": ("GatherCoverage", {}),
        }
        evaluator = SimpleNamespace(_dataframe=frame, origin_id_map={"even": 0} if rank == 0 else {"odd": 1},
                                    metrics_metamap=metrics, finalize=lambda: None)
        result = summarize_distributed_evaluator(evaluator, "valid", torch.device("cpu"))
        result["loss"] = mean_epoch_loss([(torch.tensor(float(i)), float(i+1)) for i in selected],
                                         torch.device("cpu"), distributed=True)
        torch.save(result, Path(output) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_metrics_match_unique_global_traces(tmp_path):
    mp.spawn(distributed_worker, args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)), nprocs=2, join=True)
    a, b = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in (0, 1)]
    assert a == b
    assert a["valid/HitRate1px"] == pytest.approx(0.2)
    assert a["valid/MeanAbsoluteError"] == pytest.approx(3.2)
    assert a["valid/RootMeanSquaredError"] == pytest.approx((114/5)**0.5)
    assert a["valid/GatherCoverage"] == pytest.approx(0.8)
    assert a["loss"] == pytest.approx(40/15)
    assert a["even/valid/MeanAbsoluteError"] == pytest.approx(4.0)
    assert a["odd/valid/MeanAbsoluteError"] == pytest.approx(2.0)


def test_loss_weighting_ignores_padding():
    from seismic_utils.validation import loss_weight
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    targets = torch.tensor([[[0, 1, -1]], [[1, -1, -1]]])
    assert loss_weight(loss, targets) == 3
    assert mean_epoch_loss([(1.0, 3), (9.0, 1), (float("nan"), 0)]) == 3


def test_lightning_ddp_validation_matches_single_process(tiny_hdf5, tmp_path):
    import pytorch_lightning as pl
    from conftest import make_model, make_parser
    from train.fbp_train import build_loaders
    from hardpicks.data.fbp.gather_wrappers import ShotLineGatherSubset

    parser = ShotLineGatherSubset(make_parser(tiny_hdf5), list(range(5)))
    _, loader = build_loaders(parser, parser, 2, 0)
    model = make_model()
    single = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    expected = single.validate(model, dataloaders=loader, verbose=False)[0]
    distributed = pl.Trainer(accelerator="cpu", devices=2, strategy="ddp_fork", logger=False,
                             enable_checkpointing=False, enable_progress_bar=False,
                             enable_model_summary=False, default_root_dir=str(tmp_path))
    actual = distributed.validate(model, dataloaders=loader, verbose=False)[0]
    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-6)