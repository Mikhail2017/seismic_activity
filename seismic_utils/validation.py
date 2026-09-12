"""Repeatable validation indexing, weighted losses, and exact distributed totals."""

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset


class IndexedValidationDataset(Dataset):
    """Keep original indices so DistributedSampler padding can be excluded."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "_eval_index": index}


def unique_validation_batch(batch):
    """Remove only padding duplicates from a non-shuffled DistributedSampler.

    All ranks still forward the same number of batches, avoiding DDP collective
    deadlocks. Filtering is done afterwards for loss/metric calculation.
    """
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return batch, None
    if "_eval_index" not in batch:
        raise RuntimeError("Distributed validation requires IndexedValidationDataset and shuffle=False")
    keep = batch["_eval_index"] % dist.get_world_size() == dist.get_rank()
    indices = keep.nonzero().flatten().tolist()
    size = int(batch["batch_size"])
    filtered = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim and value.shape[0] == size:
            filtered[key] = value[keep]
        elif isinstance(value, (list, tuple)) and len(value) == size:
            filtered[key] = [value[i] for i in indices]
        else:
            filtered[key] = value
    filtered["batch_size"] = len(indices)
    return filtered, keep


def loss_weight(loss_fn, targets):
    """Denominator for mean cross-entropy; gather weighting for other losses."""
    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        if loss_fn.reduction != "mean":
            raise ValueError("Validation requires cross-entropy reduction='mean'")
        valid = targets != loss_fn.ignore_index
        if loss_fn.weight is not None:
            return float(loss_fn.weight[targets[valid]].sum().detach().cpu())
        return float(valid.sum().detach().cpu())
    return float(targets.shape[0])


def mean_epoch_loss(losses, device=None, distributed=False):
    numerator = denominator = 0.0
    for item in losses:
        loss, weight = item if isinstance(item, tuple) else (item, 1.0)
        if weight > 0:
            numerator += float(loss) * weight
            denominator += weight
    totals = torch.tensor([numerator, denominator], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(totals)
    return float(totals[0] / totals[1]) if totals[1] > 0 else float("nan")


def summarize_distributed_evaluator(evaluator, prefix, device):
    """Reduce sufficient statistics, never averages of per-rank RMSE/hit rates."""
    evaluator.finalize()
    frame = evaluator._dataframe
    local_categories = list(evaluator.origin_id_map)
    categories = [None] * dist.get_world_size()
    dist.all_gather_object(categories, local_categories)
    names = sorted({name for group in categories for name in group})
    if len(names) == 1:
        names = []  # match FBPEvaluator.get_categories for single-site validation
    results = {}
    for category in [None, *names]:
        if category is None:
            selected = frame
        else:
            origin_id = evaluator.origin_id_map.get(category)
            selected = frame[frame["OriginId"] == origin_id] if origin_id is not None else frame.iloc[:0]
        errors = selected["Errors"].dropna().to_numpy(dtype=np.float64) if "Errors" in selected else np.array([])
        values = [len(errors), np.abs(errors).sum(), errors.sum(), np.square(errors).sum()]
        metrics = list(evaluator.metrics_metamap.items())
        for _, (kind, params) in metrics:
            if kind == "HitRate":
                values.append(float((np.abs(errors) < params["buffer_size_px"]).sum()))
        values.extend([
            float(selected["GatherCoverage"].sum()) if "GatherCoverage" in selected else 0.0,
            float(selected["ExpectedCoverage"].sum()) if "ExpectedCoverage" in selected else 0.0,
        ])
        totals = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(totals)
        count, absolute, signed, squared = totals[:4].tolist()
        hit_index = 4
        for name, (kind, _) in metrics:
            value = None
            if kind == "HitRate":
                value = float(totals[hit_index]) / count if count else None
                hit_index += 1
            elif kind == "MeanAbsoluteError" and count:
                value = absolute / count
            elif kind == "MeanBiasError" and count:
                value = signed / count
            elif kind == "RootMeanSquaredError" and count:
                value = (squared / count) ** 0.5
            elif kind == "GatherCoverage" and totals[-1] > 0:
                value = float(totals[-2] / totals[-1])
            if value is not None:
                key = f"{category}/{prefix}/{name}" if category is not None else f"{prefix}/{name}"
                results[key] = value
    return results