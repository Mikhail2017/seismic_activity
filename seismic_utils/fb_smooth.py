"""Horizon (before/after) first-break extraction.

Faithful numpy/torch port of ``_fb_smooth_result`` from geo-stack
``first_break_picking`` (not imported). Class map: ``0`` = before first break,
``1`` = at/after. The legacy rule compares two window sums; it does not
guarantee a contiguous run of ``threshold`` after-class samples. It is kept
for checkpoint compatibility rather than silently changing the picker.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SMOOTH_THRESHOLD = 50
_BAD_PICK = 0  # hardpicks BAD_FIRST_BREAK_PICK_INDEX


def fb_smooth_result(
    predicted: np.ndarray,
    *,
    threshold: int = DEFAULT_SMOOTH_THRESHOLD,
) -> np.ndarray:
    """Extract per-trace first-break sample indices from a before/after class map.

    Parameters
    ----------
    predicted
        Integer class map shaped ``(n_samples, n_traces)``. Class ``1`` is after-break.
    threshold
        Window length (samples) used to test that the after-class run is stable.

    Returns
    -------
    picks
        ``(n_traces,)`` int64 sample indices. ``0`` means no pick (empty/unstable run).
    """
    arr = np.asarray(predicted)
    if arr.ndim != 2:
        raise ValueError(f"predicted must be (n_samples, n_traces), got {arr.shape}")
    n_samples, n_traces = arr.shape
    del n_samples
    picks = np.full(n_traces, _BAD_PICK, dtype=np.int64)
    thr = int(threshold)
    if thr < 1:
        raise ValueError("threshold must be positive")
    for i in range(n_traces):
        col = arr[:, i]
        ones = np.flatnonzero(col == 1)
        if ones.size < 2:
            continue
        count = 0
        n_ones = int(ones.size)
        while count + 1 < n_ones:
            start_a = int(ones[count])
            start_b = int(ones[count + 1])
            a = int(col[start_a : start_a + thr].sum())
            b = int(col[start_b : start_b + thr].sum())
            if a == b:
                picks[i] = start_a
                break
            count += 1
    return picks


def fb_smooth_from_logits(raw_preds, *, threshold: int = DEFAULT_SMOOTH_THRESHOLD, sample_counts=None):
    """Decode UNet logits ``(B, C, n_traces, n_samples)`` → pick indices + after-class prob.

    Returns ``(picks, probabilities)`` as tensors on the same device as *raw_preds*.
    Picks use ``0`` for no-pick; those probabilities are ``NaN``.
    """
    import torch

    if not torch.is_tensor(raw_preds):
        raise TypeError(f"raw_preds must be a tensor, got {type(raw_preds)!r}")
    if raw_preds.ndim != 4:
        raise ValueError(f"raw_preds must be (B, C, traces, samples), got {tuple(raw_preds.shape)}")

    probs = torch.softmax(raw_preds, dim=1)
    class_map = probs.argmax(dim=1)  # (B, n_traces, n_samples)
    batch_size, n_traces, n_samples = class_map.shape
    pred_np = class_map.detach().cpu().numpy()
    pick_np = np.zeros((batch_size, n_traces), dtype=np.int64)
    if sample_counts is None:
        counts = [n_samples] * batch_size
    else:
        counts = torch.as_tensor(sample_counts).reshape(-1).tolist()
        if len(counts) != batch_size or any(int(n) <= 0 or int(n) > n_samples for n in counts):
            raise ValueError("sample_counts must contain the unpadded length of every gather")
    for b in range(batch_size):
        pick_np[b] = fb_smooth_result(pred_np[b, :, :int(counts[b])].T, threshold=threshold)

    device = raw_preds.device
    picks = torch.from_numpy(pick_np).to(device=device, dtype=torch.long)
    after = probs[:, -1]  # (B, n_traces, n_samples)
    gather_idx = picks.unsqueeze(-1).clamp(0, max(n_samples - 1, 0))
    pick_prob = after.gather(2, gather_idx).squeeze(-1)
    invalid = picks <= 0
    pick_prob = pick_prob.masked_fill(invalid, float("nan"))
    return picks, pick_prob
