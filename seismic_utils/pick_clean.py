"""Lateral (along-line) cleaner for predicted first-break sample indices.

Predictions only — never rewrite labels. Flag traces whose pick jumps away from
a robust Theil–Sen fit of pick vs offset (local median if offsets are constant),
then replace those picks by interpolating from the remaining anchors. Gathers
where too many valid picks are flagged are left unchanged (systematic early/late
picks, not one-trace crashes).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

DEFAULT_LATERAL_WINDOW = 15
DEFAULT_LATERAL_MAX_DEV = 15.0
DEFAULT_LATERAL_MAX_FLAG_FRAC = 0.30
DEFAULT_LATERAL_MIN_ANCHORS = 3


def clean_picks_lateral(
    picks: np.ndarray,
    offsets: np.ndarray | None = None,
    *,
    window: int = DEFAULT_LATERAL_WINDOW,
    max_dev_samples: float = DEFAULT_LATERAL_MAX_DEV,
    max_flag_frac: float = DEFAULT_LATERAL_MAX_FLAG_FRAC,
    min_anchors: int = DEFAULT_LATERAL_MIN_ANCHORS,
) -> tuple[np.ndarray, int]:
    """Replace isolated pick outliers using neighboring traces.

    Parameters
    ----------
    picks
        Per-trace sample indices. ``<= 0`` means no pick.
    offsets
        Shot-receiver offsets, same length as *picks*. Trace index is used when
        offsets are missing or constant.

    Returns
    -------
    cleaned, n_replaced
        Integer pick array and how many traces were rewritten.
    """
    out = np.asarray(picks, dtype=np.int64).reshape(-1).copy()
    n = int(out.size)
    if n == 0:
        return out, 0
    window = int(window)
    if window < 1:
        raise ValueError("window must be positive")
    max_dev = float(max_dev_samples)
    if max_dev <= 0:
        raise ValueError("max_dev_samples must be positive")
    min_anchors = int(min_anchors)
    if min_anchors < 2:
        raise ValueError("min_anchors must be >= 2")

    valid = out > 0
    n_valid = int(valid.sum())
    if n_valid < min_anchors:
        return out, 0

    flagged = _flag_outliers(out, offsets, valid, window=window, max_dev=max_dev)

    n_flag_valid = int(flagged[valid].sum())
    if n_valid and (n_flag_valid / n_valid) > float(max_flag_frac):
        return out, 0

    anchors = valid & ~flagged
    if int(anchors.sum()) < min_anchors:
        return out, 0

    to_replace = flagged | ~valid
    if not np.any(to_replace):
        return out, 0

    x = _axis(offsets, n, anchors)
    xa = x[anchors]
    ya = out[anchors].astype(np.float64)
    order = np.argsort(xa, kind="mergesort")
    xa = xa[order]
    ya = ya[order]
    xu, inv = np.unique(xa, return_inverse=True)
    if xu.size < 2:
        return out, 0
    yu = np.zeros(xu.size, dtype=np.float64)
    counts = np.zeros(xu.size, dtype=np.float64)
    np.add.at(yu, inv, ya)
    np.add.at(counts, inv, 1.0)
    yu /= np.maximum(counts, 1.0)

    from scipy.interpolate import interp1d

    fn = interp1d(xu, yu, kind="linear", fill_value="extrapolate", assume_sorted=True)
    lo_s = 1
    hi_s = max(int(np.max(out[anchors])), 1)
    repl = np.rint(fn(x[to_replace])).astype(np.int64)
    repl = np.clip(repl, lo_s, hi_s)
    changed = repl != out[to_replace]
    out[to_replace] = repl
    return out, int(changed.sum())


def apply_lateral_clean_to_frame(
    df: pd.DataFrame,
    *,
    window: int = DEFAULT_LATERAL_WINDOW,
    max_dev_samples: float = DEFAULT_LATERAL_MAX_DEV,
    max_flag_frac: float = DEFAULT_LATERAL_MAX_FLAG_FRAC,
    min_anchors: int = DEFAULT_LATERAL_MIN_ANCHORS,
) -> tuple[pd.DataFrame, int]:
    """Rewrite ``Predictions`` (and ``Errors``) gather-by-gather. Labels are not touched."""
    if df.empty or "Predictions" not in df.columns:
        return df, 0
    keys = [c for c in ("OriginId", "GatherId", "ShotId") if c in df.columns]
    if not keys:
        return df, 0
    # Eval concatenates per-batch frames with a repeating RangeIndex. ``loc`` on
    # group labels then pulls extra rows. Reset so group index == row position.
    out = df.reset_index(drop=True).copy()
    pred_col = out.columns.get_loc("Predictions")
    err_col = out.columns.get_loc("Errors") if "Errors" in out.columns else None
    total = 0
    for _, loc in out.groupby(keys, dropna=False, sort=False):
        pos = loc.index.to_numpy(dtype=np.intp)
        picks = pd.to_numeric(loc["Predictions"], errors="coerce").to_numpy(dtype=np.float64)
        picks = np.where(np.isfinite(picks), picks, 0).astype(np.int64)
        offsets = None
        if "Offset" in loc.columns:
            raw = pd.to_numeric(loc["Offset"], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(raw).any():
                offsets = raw
        cleaned, n_rep = clean_picks_lateral(
            picks,
            offsets,
            window=window,
            max_dev_samples=max_dev_samples,
            max_flag_frac=max_flag_frac,
            min_anchors=min_anchors,
        )
        if n_rep == 0:
            continue
        total += n_rep
        out.iloc[pos, pred_col] = cleaned
        if err_col is not None:
            err = pd.to_numeric(out.iloc[pos, err_col], errors="coerce").to_numpy(dtype=np.float64)
            out.iloc[pos, err_col] = err + (cleaned.astype(np.float64) - picks.astype(np.float64))
    return out, total


def lateral_clean_kwargs(args: Mapping[str, Any] | None) -> dict[str, Any]:
    data = dict(args or {})
    return {
        "window": int(data.get("lateral_window", DEFAULT_LATERAL_WINDOW)),
        "max_dev_samples": float(data.get("lateral_max_dev", DEFAULT_LATERAL_MAX_DEV)),
        "max_flag_frac": float(data.get("lateral_max_flag_frac", DEFAULT_LATERAL_MAX_FLAG_FRAC)),
        "min_anchors": int(data.get("lateral_min_anchors", DEFAULT_LATERAL_MIN_ANCHORS)),
    }


def _flag_outliers(
    picks: np.ndarray,
    offsets: np.ndarray | None,
    valid: np.ndarray,
    *,
    window: int,
    max_dev: float,
) -> np.ndarray:
    """Theil–Sen vs offset when possible; else local median along the line."""
    n = int(picks.size)
    x = _axis(offsets, n, valid)
    if np.unique(x[valid]).size >= 2 and int(valid.sum()) >= 3:
        from scipy.stats import theilslopes

        slope, intercept, lo_s, hi_s = theilslopes(
            picks[valid].astype(np.float64), x[valid]
        )
        del lo_s, hi_s
        fitted = intercept + slope * x
        return valid & (np.abs(picks.astype(np.float64) - fitted) > max_dev)

    half = window // 2
    flagged = np.zeros(n, dtype=bool)
    for i in range(n):
        if not valid[i]:
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neigh = picks[lo:hi]
        neigh = neigh[neigh > 0]
        if neigh.size == 0:
            continue
        if abs(float(picks[i]) - float(np.median(neigh))) > max_dev:
            flagged[i] = True
    return flagged


def _axis(offsets: np.ndarray | None, n: int, anchors: np.ndarray) -> np.ndarray:
    if offsets is not None:
        x = np.asarray(offsets, dtype=np.float64).reshape(-1)
        if x.size == n and np.isfinite(x[anchors]).sum() >= 2 and np.unique(x[anchors]).size >= 2:
            return np.where(np.isfinite(x), x, np.arange(n, dtype=np.float64))
    return np.arange(n, dtype=np.float64)
