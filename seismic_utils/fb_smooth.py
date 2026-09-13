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
BEFORE_AFTER_DECODERS = ("legacy", "change_point")
DEFAULT_BEFORE_AFTER_DECODER = "legacy"
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


def normalize_before_after_decoder(decoder: str | None) -> str:
    """Missing checkpoint metadata must preserve the legacy selection rule."""
    name = DEFAULT_BEFORE_AFTER_DECODER if decoder is None else decoder
    if name not in BEFORE_AFTER_DECODERS:
        raise ValueError(f"Unknown before/after decoder {name!r}; expected {BEFORE_AFTER_DECODERS}")
    return name


def fb_change_point_from_logits(raw_preds, *, sample_counts=None):
    """Find the maximum-likelihood single before→after transition per trace.

    Input is floating logits ``(B, 2, traces, time)``: class 0 is before and
    class 1 is at/after. For each real length T, score boundaries k=0..T with
    ``C(k) = -sum(log P_before[:k]) - sum(log P_after[k:T])``.
    The equivalent relative cost ``C(k)-C(0)`` is a prefix sum of log odds;
    this avoids subtracting two large full-trace costs and needs O(T) work.

    Both endpoints represent no observed transition (all-after/all-before).
    An interior minimum must strictly beat both endpoints; otherwise return
    0. Tied interior minima select the earliest boundary. Non-finite real
    samples invalidate their trace; time padding never participates in scoring.

    Returns integer picks and float32 after-class probabilities at those picks,
    both shaped ``(B, traces)`` on the input device. No-pick probabilities are
    NaN. This probability is NOT calibrated boundary confidence. Scoring uses
    float32 even for mixed-precision logits and does not build a gradient graph.
    """
    import torch

    if not torch.is_tensor(raw_preds):
        raise TypeError(f"raw_preds must be a tensor, got {type(raw_preds)!r}")
    if raw_preds.ndim != 4 or raw_preds.shape[1] != 2 or raw_preds.shape[-1] < 1:
        raise ValueError("raw_preds must be (B, 2, traces, samples) with samples > 0")
    if not raw_preds.is_floating_point():
        raise TypeError("raw_preds must contain floating-point logits")
    batch_size, _, n_traces, n_samples = raw_preds.shape
    counts = [n_samples] * batch_size if sample_counts is None else torch.as_tensor(sample_counts).reshape(-1).tolist()
    if len(counts) != batch_size or any(
        not np.isfinite(n) or n < 1 or n > n_samples or int(n) != n for n in counts
    ):
        raise ValueError("sample_counts must contain the integer unpadded length of every gather")

    with torch.no_grad():
        picks = torch.zeros((batch_size, n_traces), dtype=torch.long, device=raw_preds.device)
        probabilities = torch.full((batch_size, n_traces), float("nan"), device=raw_preds.device, dtype=torch.float32)
        for b, count in enumerate(counts):
            count = int(count)
            # Slice BEFORE softmax/cumulative sums, including non-finite padding.
            logits = raw_preds[b, :, :, :count].float()
            log_probs = torch.log_softmax(logits, dim=0)
            finite = torch.isfinite(log_probs).all(dim=0).all(dim=-1)
            log_odds = log_probs[1] - log_probs[0]
            relative_cost = torch.cat(
                [log_odds.new_zeros((n_traces, 1)), log_odds.cumsum(dim=-1)], dim=-1
            )
            finite = finite & torch.isfinite(relative_cost).all(dim=-1)
            best_cost, best = relative_cost.min(dim=-1)
            endpoint_cost = torch.minimum(relative_cost[:, 0], relative_cost[:, -1])
            valid = finite & (best > 0) & (best < count) & (best_cost < endpoint_cost)
            picks[b] = torch.where(valid, best, 0)
            prob = log_probs[1].gather(-1, picks[b].unsqueeze(-1)).squeeze(-1).exp()
            probabilities[b] = prob.masked_fill(~valid, float("nan"))
    return picks, probabilities


def decode_before_after_logits(raw_preds, *, decoder=None, threshold=DEFAULT_SMOOTH_THRESHOLD, sample_counts=None):
    """Shared before/after decoding; ``threshold`` is used only by legacy."""
    decoder = normalize_before_after_decoder(decoder)
    if decoder == "change_point":
        return fb_change_point_from_logits(raw_preds, sample_counts=sample_counts)
    return fb_smooth_from_logits(raw_preds, threshold=threshold, sample_counts=sample_counts)
