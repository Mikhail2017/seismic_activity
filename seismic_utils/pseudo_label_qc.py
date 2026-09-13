"""Offset-bin 2σ quality control for self-training pseudo-labels.

Admit/reject only. Does not interpolate or import ``pick_clean``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

N_OFFSET_BINS = 20
N_SIGMA = 2.0
MIN_SURVIVE_FRAC = 0.85


def _as_1d(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def qc_gather_picks(
    offsets: np.ndarray,
    picks: np.ndarray,
    *,
    n_bins: int = N_OFFSET_BINS,
    n_sigma: float = N_SIGMA,
    min_survive_frac: float = MIN_SURVIVE_FRAC,
) -> dict[str, Any]:
    """Flag per-trace outliers and decide whether to admit the gather.

    Rules:
    - no pick / non-finite offset → not surviving
    - bin with fewer than 2 picks → no 2σ reject for those traces
    - σ = 0 → keep (deviation 0)
    - admit iff survivors / all traces ≥ ``min_survive_frac``
    """
    off = _as_1d(offsets)
    pred = _as_1d(picks)
    n = int(max(off.size, pred.size))
    if off.size < n:
        off = np.pad(off, (0, n - off.size), constant_values=np.nan)
    if pred.size < n:
        pred = np.pad(pred, (0, n - pred.size), constant_values=np.nan)

    has_pick = np.isfinite(pred) & (pred > 0)
    has_off = np.isfinite(off)
    eligible = has_pick & has_off
    survive = np.zeros(n, dtype=bool)
    survive[eligible] = True

    if int(eligible.sum()) >= 2 and n_bins >= 1:
        lo = float(np.nanmin(off[has_off])) if np.any(has_off) else 0.0
        hi = float(np.nanmax(off[has_off])) if np.any(has_off) else 1.0
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, int(n_bins) + 1)
        bin_id = np.digitize(off, edges[1:-1], right=False)
        bin_id = np.clip(bin_id, 0, int(n_bins) - 1)
        for b in range(int(n_bins)):
            in_bin = eligible & (bin_id == b)
            idx = np.where(in_bin)[0]
            if idx.size < 2:
                continue
            vals = pred[idx]
            mean = float(vals.mean())
            std = float(vals.std(ddof=0))
            if std == 0.0:
                continue
            deviate = np.abs(vals - mean) > (float(n_sigma) * std)
            survive[idx[deviate]] = False

    n_surv = int(survive.sum())
    frac = float(n_surv) / float(n) if n else 0.0
    admit = frac >= float(min_survive_frac)
    cleaned = np.full(n, np.nan, dtype=np.float64)
    if admit:
        cleaned[survive] = pred[survive]
    return {
        "admit": bool(admit),
        "n_traces": n,
        "n_survive": n_surv,
        "survive_frac": frac,
        "survive": survive,
        "picks": cleaned,
    }


def draw_without_replacement(
    remaining: Sequence[Any],
    n: int,
    rng: np.random.Generator,
) -> tuple[list[Any], list[Any]]:
    pool = list(remaining)
    take = min(int(n), len(pool))
    if take <= 0:
        return [], pool
    order = rng.permutation(len(pool))
    drawn_idx = set(int(i) for i in order[:take])
    drawn = [pool[i] for i in range(len(pool)) if i in drawn_idx]
    rest = [pool[i] for i in range(len(pool)) if i not in drawn_idx]
    return drawn, rest
