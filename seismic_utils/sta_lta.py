"""Adaptive STA-LTA with outlier statistics (Jones & van der Baan, 2015).

Python port of `STA-LTA-OS` (`hmmscan` / `exmax` / `sop` / `es94ap` / `genprep`).
The original MATLAB is written for multi-channel borehole event detection; this
module keeps that scan and adds a first-break wrapper that runs the same HMM on
each trace independently and returns the first P-pick.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from .dataset import ShotGather

_EPS = 1e-12
_DEAD_TRACE_EPS = 1e-8

# Paper defaults are for borehole data at 4 kHz (Lw=2000, Sw=200 samples).
_REF_FS_HZ = 4000.0
_DEFAULT_LW_S = 2000.0 / _REF_FS_HZ  # 0.50 s
_DEFAULT_SW_S = 200.0 / _REF_FS_HZ  # 0.05 s
_DEFAULT_EOS_S = 1000.0 / _REF_FS_HZ  # 0.25 s


@dataclass(frozen=True)
class StaLtaOptions:
    """Controlling parameters for STA-LTA-OS (seconds unless noted)."""

    lw_s: float = _DEFAULT_LW_S
    sw_s: float = _DEFAULT_SW_S
    th: float = 1.3
    p_th: float = 2.0
    cf: str = "env"
    f0: float = 0.10
    fw: tuple[float, float] = (0.005, 0.4)
    slv: str = "exp"
    n_iter: int = 100
    tol: float = -1.0
    weed: bool = False
    nstd: float = 3.0
    qth: float = 0.1
    tmax: float = 0.95
    eos_s: float = _DEFAULT_EOS_S
    onset_pick: bool = True
    min_lw: int = 32
    min_sw: int = 8
    s0: int = 1


def default_sta_lta_options(**overrides: Any) -> StaLtaOptions:
    """Return options with *overrides* applied (unknown keys are ignored)."""
    base = StaLtaOptions()
    valid = {f.name for f in StaLtaOptions.__dataclass_fields__.values()}
    return replace(base, **{k: v for k, v in overrides.items() if k in valid})


def options_from_mapping(raw: Mapping[str, Any] | None) -> StaLtaOptions:
    if not raw:
        return StaLtaOptions()
    return default_sta_lta_options(**dict(raw))


def _as_column_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def genprep(x: np.ndarray, rules: Sequence[Any], fs: float | None = None) -> np.ndarray:
    """Generic preprocessing of column-vector traces (MATLAB ``genprep``)."""
    xf = _as_column_matrix(x)
    n = len(rules)
    j = 0
    while j < n:
        rule = rules[j]
        if not isinstance(rule, str):
            j += 1
            continue
        name = rule.lower()
        if name == "demean":
            xf = xf - np.mean(xf, axis=0, keepdims=True)
        elif name == "detrend":
            xf = signal.detrend(xf, axis=0, type="linear")
        elif name == "env":
            xf = np.abs(signal.hilbert(xf, axis=0)).astype(np.float64, copy=False)
        elif name == "abs":
            xf = np.abs(xf)
        elif name == "pow":
            power = float(rules[j + 1]) if j + 1 < n else 2.0
            xf = np.abs(xf) ** power
            j += 1
        elif name == "mir":
            xf = np.vstack([xf, xf[::-1]])
        elif name == "filt":
            if fs is None:
                raise ValueError("genprep 'filt' requires sampling frequency fs")
            corners = np.atleast_1d(np.asarray(rules[j + 1], dtype=np.float64))
            frule = str(rules[j + 2]).lower() if j + 2 < n and isinstance(rules[j + 2], str) else ""
            wn = corners * 2.0 / float(fs)
            if frule in {"", "lowpass", "low"}:
                b, a = signal.butter(2, wn, btype="lowpass" if wn.size == 1 else "bandpass")
            else:
                b, a = signal.butter(2, wn, btype=frule)
            xf = signal.filtfilt(b, a, xf, axis=0)
            j += 2
        j += 1
    return xf


def hmm_init(x: np.ndarray, f: float | np.ndarray, sort_spec: Sequence[str] | None = None) -> np.ndarray:
    """Initial 2-state HMM parameters from the outermost ``f`` fraction of *x*."""
    x = _as_column_matrix(x)
    n_x, n_c = x.shape
    frac = np.broadcast_to(np.asarray(f, dtype=np.float64).reshape(-1), (n_c,))
    n_out = np.clip(np.round(frac * n_x).astype(int), 1, max(n_x - 1, 1))

    kind, order = ("abs", "descend")
    if sort_spec:
        kind = str(sort_spec[0]).lower()
        if len(sort_spec) > 1:
            order = str(sort_spec[1]).lower()

    if kind == "abs":
        x0 = np.abs(x)
    elif kind == "sq":
        x0 = x ** 2
    else:
        x0 = x.copy()
    x0 = np.sort(x0, axis=0)
    if order.startswith("desc"):
        x0 = x0[::-1]

    q = np.zeros((4, n_c), dtype=np.float64)
    for c in range(n_c):
        k = int(n_out[c])
        out = x0[:k, c]
        null = x0[k:, c]
        q[0, c] = float(np.mean(out))
        q[1, c] = float(np.mean(null)) if null.size else q[0, c]
        q[2, c] = float(np.var(out)) if out.size > 1 else 0.0
        q[3, c] = float(np.var(null)) if null.size > 1 else 0.0
    return q


def _get_a(p1: np.ndarray, p0: np.ndarray, f: np.ndarray) -> np.ndarray:
    n_x = p1.shape[0]
    ff = np.broadcast_to(np.asarray(f, dtype=np.float64).reshape(1, -1), (n_x, p1.shape[1]))
    num = ff * p1
    den = num + (1.0 - ff) * p0
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(den > _EPS, num / den, 0.0)
    return np.clip(a, 0.0, 1.0)


def _exstep(
    slv: str,
    x: np.ndarray,
    f: np.ndarray,
    m1: np.ndarray,
    m0: np.ndarray,
    v1: np.ndarray,
    v0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_x = x.shape[0]
    slv = slv.lower()
    m1 = np.maximum(m1, _EPS)
    m0 = np.maximum(m0, _EPS)
    v1 = np.maximum(v1, _EPS)
    v0 = np.maximum(v0, _EPS)

    if slv == "gauss":
        s1 = 1.0 / np.sqrt(2.0 * v1)
        s0 = 1.0 / np.sqrt(2.0 * v0)
        p1 = (s1 / np.sqrt(np.pi)) * np.exp(-((x - m1) ** 2) * (s1 ** 2))
        p0 = (s0 / np.sqrt(np.pi)) * np.exp(-((x - m0) ** 2) * (s0 ** 2))
        a = _get_a(p1, p0, f)
        like = (
            np.sum(a, axis=0) * np.log(np.maximum(s1 * f / np.sqrt(np.pi), _EPS))
            + np.sum(1.0 - a, axis=0) * np.log(np.maximum(s0 * (1.0 - f) / np.sqrt(np.pi), _EPS))
            - (s1 ** 2) * np.sum(a * (x - m1) ** 2, axis=0)
            - (s0 ** 2) * np.sum((1.0 - a) * (x - m0) ** 2, axis=0)
        )
        return like, a

    if slv == "zgauss":
        s1 = 1.0 / np.sqrt(2.0 * v1)
        s0 = 1.0 / np.sqrt(2.0 * v0)
        p1 = (s1 / np.sqrt(np.pi)) * np.exp(-(x ** 2) * (s1 ** 2))
        p0 = (s0 / np.sqrt(np.pi)) * np.exp(-(x ** 2) * (s0 ** 2))
        a = _get_a(p1, p0, f)
        like = (
            np.sum(a, axis=0) * np.log(np.maximum(s1 * f / np.sqrt(np.pi), _EPS))
            + np.sum(1.0 - a, axis=0) * np.log(np.maximum(s0 * (1.0 - f) / np.sqrt(np.pi), _EPS))
            - (s1 ** 2) * np.sum(a * (x ** 2), axis=0)
            - (s0 ** 2) * np.sum((1.0 - a) * (x ** 2), axis=0)
        )
        return like, a

    if slv == "ray":
        s1 = 1.0 / m1
        s0 = 1.0 / m0
        p1 = x * s1 * np.exp(-0.5 * s1 * (x ** 2))
        p0 = x * s0 * np.exp(-0.5 * s0 * (x ** 2))
        a = _get_a(p1, p0, f)
        like = (
            np.sum(a, axis=0) * np.log(np.maximum(f * s1, _EPS))
            + np.sum(a * np.log(np.maximum(x, _EPS)), axis=0)
            - 0.5 * s1 * np.sum(a * (x ** 2), axis=0)
            + np.sum(1.0 - a, axis=0) * np.log(np.maximum((1.0 - f) * s0, _EPS))
            + np.sum((1.0 - a) * np.log(np.maximum(x, _EPS)), axis=0)
            - 0.5 * s0 * np.sum((1.0 - a) * (x ** 2), axis=0)
        )
        return like, a

    im1 = 1.0 / m1
    im0 = 1.0 / m0
    p1 = np.exp(-x * im1) * im1
    p0 = np.exp(-x * im0) * im0
    a = _get_a(p1, p0, f)
    like = (
        np.log(np.maximum(f * im1, _EPS)) * np.sum(a, axis=0)
        - np.sum(a * x, axis=0) * im1
        + np.log(np.maximum((1.0 - f) * im0, _EPS)) * np.sum(1.0 - a, axis=0)
        - np.sum((1.0 - a) * x, axis=0) * im0
    )
    return like, a


def _em_get_th(means_vars: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m1 = f * means_vars[0] + (1.0 - f) * means_vars[1]
    m2 = f * means_vars[2] + (1.0 - f) * means_vars[3]
    th1 = np.sqrt((((means_vars[0] - m1) ** 2) + ((means_vars[1] - m1) ** 2)) / np.maximum(m1 ** 2, _EPS))
    th2 = np.sqrt((((means_vars[2] - m1) ** 2) + ((means_vars[3] - m1) ** 2)) / np.maximum(m2 ** 2, _EPS))
    return th1, th2


def exmax(
    x: np.ndarray,
    *,
    slv: str = "exp",
    f0: float | np.ndarray = 0.10,
    q0: np.ndarray | None = None,
    n_iter: int = 100,
    tol: float = -1.0,
    weed: bool = False,
    nstd: float = 3.0,
    cf: str = "env",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Two-state EM classification. Returns ``(P, f, Q)`` for converged columns."""
    del cf  # only used with the unpublished MinThr.mat weeder in MATLAB
    x_all = _as_column_matrix(x)
    n_x, n_c = x_all.shape
    if n_x < 4 or n_c == 0:
        empty_q = {"params": np.zeros((4, 0)), "key": np.zeros((0,), dtype=int)}
        return np.zeros((n_x, 0)), np.zeros((0,)), empty_q

    work = x_all.copy()
    f = np.broadcast_to(np.asarray(f0, dtype=np.float64).reshape(-1), (n_c,)).copy()
    if q0 is None:
        q0 = hmm_init(work, f)
    m1 = q0[0].astype(np.float64, copy=True)
    m0 = q0[1].astype(np.float64, copy=True)
    v1 = q0[2].astype(np.float64, copy=True)
    v0 = q0[3].astype(np.float64, copy=True)
    idx = np.arange(n_c, dtype=int)
    conv_tol = 1.0e-6 * n_x if tol < 0 else float(tol)

    a_final: list[np.ndarray] = []
    f_final: list[float] = []
    m1_f: list[float] = []
    m0_f: list[float] = []
    v1_f: list[float] = []
    v0_f: list[float] = []
    key: list[int] = []

    like0 = np.zeros(n_c, dtype=np.float64)
    for nn in range(int(n_iter)):
        if work.shape[1] == 0:
            break
        like, a = _exstep(slv, work, f, m1, m0, v1, v0)
        if nn > 0:
            done = np.where((like - like0) < conv_tol)[0]
            if done.size:
                for k in done.tolist():
                    key.append(int(idx[k]))
                    a_final.append(a[:, k].copy())
                    f_final.append(float(f[k]))
                    m1_f.append(float(m1[k]))
                    m0_f.append(float(m0[k]))
                    v1_f.append(float(v1[k]))
                    v0_f.append(float(v0[k]))
                keep = np.ones(work.shape[1], dtype=bool)
                keep[done] = False
                work = work[:, keep]
                a = a[:, keep]
                f = f[keep]
                m1 = m1[keep]
                m0 = m0[keep]
                v1 = v1[keep]
                v0 = v0[keep]
                like = like[keep]
                idx = idx[keep]
            if work.shape[1] == 0 or np.all(~np.isfinite(like)):
                break

        if work.shape[1] == 0:
            break
        sum_a = np.maximum(np.sum(a, axis=0), _EPS)
        sum_1a = np.maximum(np.sum(1.0 - a, axis=0), _EPS)
        f = np.clip(np.sum(a, axis=0) / n_x, _EPS, 1.0 - _EPS)
        if slv.lower() == "ray":
            m1 = 0.5 * np.sum(a * (work ** 2), axis=0) / sum_a
            m0 = 0.5 * np.sum((1.0 - a) * (work ** 2), axis=0) / sum_1a
        else:
            m1 = np.sum(a * work, axis=0) / sum_a
            m0 = np.sum((1.0 - a) * work, axis=0) / sum_1a
        v1 = np.sum(a * (work - m1) ** 2, axis=0) / sum_a
        v0 = np.sum((1.0 - a) * (work - m0) ** 2, axis=0) / sum_1a
        like0 = like

    if not key:
        empty_q = {"params": np.zeros((4, 0)), "key": np.zeros((0,), dtype=int)}
        return np.zeros((n_x, 0)), np.zeros((0,)), empty_q

    order = np.argsort(np.asarray(key, dtype=int))
    a_out = np.column_stack([a_final[i] for i in order])
    f_out = np.asarray([f_final[i] for i in order], dtype=np.float64)
    params = np.vstack(
        [
            np.asarray([m1_f[i] for i in order]),
            np.asarray([m0_f[i] for i in order]),
            np.asarray([v1_f[i] for i in order]),
            np.asarray([v0_f[i] for i in order]),
        ]
    )
    keys = np.asarray([key[i] for i in order], dtype=int)

    if weed and f_out.size:
        th1, th2 = _em_get_th(params, f_out)
        th = th2 if slv.lower() in {"zgauss"} else th1
        # MATLAB loads unpublished MinThr.mat; without it, drop near-identical states.
        t0 = 0.25 + float(nstd) * 0.05
        keep = th > t0
        a_out = a_out[:, keep]
        f_out = f_out[keep]
        params = params[:, keep]
        keys = keys[keep]

    return a_out, f_out, {"params": params, "key": keys}


def _hann_filt(x: np.ndarray, n_h: int = 30) -> np.ndarray:
    x = _as_column_matrix(x)
    n_x = x.shape[0]
    n_h = max(int(n_h), 3)
    hf = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_h) / (n_h - 1)))
    hf = hf / np.sum(hf)
    padlen = 3 * (len(hf) - 1)
    work = x
    if n_x <= padlen:
        work = np.vstack([work, np.zeros((padlen - n_x + 1, work.shape[1]))])
    xs = signal.filtfilt(hf, [1.0], work, axis=0, padlen=padlen)
    return xs[:n_x]


def es94_autopick(ratio: np.ndarray, n_h: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Earle & Shearer (1994) onset pick on a subtractive STA series (column vectors)."""
    r = _as_column_matrix(ratio)
    rs = _hann_filt(r, n_h)
    n_c = r.shape[1]
    picks = np.zeros(n_c, dtype=np.float64)
    quality = np.zeros(n_c, dtype=np.float64)
    for c in range(n_c):
        pos = np.flatnonzero(rs[:, c] > 0)
        if pos.size == 0:
            continue
        t1 = int(pos[0])
        rd1 = np.diff(rs[t1:, c])
        if rd1.size < 3:
            continue
        peaks = np.flatnonzero((rd1[1:] < 0) & (rd1[:-1] > 0))
        if peaks.size == 0:
            continue
        tm1 = int(peaks[0])
        quality[c] = float(rs[t1 + tm1 + 1, c]) if (t1 + tm1 + 1) < rs.shape[0] else float(rs[t1 + tm1, c])
        if tm1 < 2:
            continue
        rd2 = np.diff(rd1[: tm1 + 1])
        if rd2.size < 3:
            continue
        up = np.flatnonzero((rd2[1:] < 0) & (rd2[:-1] > 0))
        if up.size:
            p1 = int(up[-1])
        else:
            down = np.flatnonzero((rd2[1:] > 0) & (rd2[:-1] < 0))
            if down.size == 0:
                continue
            p1 = int(down[-1])
        # MATLAB: p = 2 + t1 + p1 with 1-based t1/p1 → 0-based t1 + p1 + 3
        picks[c] = float(t1 + p1 + 3)
    return picks, quality, rs


def _window_samples(n_samples: int, fs_hz: float, opts: StaLtaOptions) -> tuple[int, int]:
    fs = max(float(fs_hz), _EPS)
    lw = int(round(opts.lw_s * fs))
    sw = int(round(opts.sw_s * fs))
    lw = max(lw, int(opts.min_lw))
    sw = max(sw, int(opts.min_sw))
    if n_samples < opts.min_lw:
        lw = max(n_samples, 4)
        sw = max(min(sw, max(lw // 4, 3)), 3)
        return lw, sw
    lw = min(lw, n_samples)
    sw = min(sw, max(lw // 4, opts.min_sw), max(n_samples // 8, opts.min_sw))
    sw = max(sw, 3)
    if sw >= lw:
        sw = max(lw // 4, 3)
    return int(lw), int(sw)


def outlier_probability_scan(
    y: np.ndarray,
    *,
    fs_hz: float,
    opts: StaLtaOptions | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill per-sample outlier probabilities with sliding-window EM (MATLAB ``hmmscan`` core)."""
    opts = opts or StaLtaOptions()
    y = _as_column_matrix(y)
    n_y, n_c = y.shape
    lw, sw = _window_samples(n_y, fs_hz, opts)
    p = np.zeros((n_y, n_c), dtype=np.float64)
    f_hat = np.full(n_c, opts.f0, dtype=np.float64)
    if n_y < 8 or n_c == 0:
        return p, f_hat

    f0 = f_hat.copy()
    q0 = hmm_init(y[:lw], f0)
    yc = 0
    step = max(lw // 2, 1)
    while yc < n_y - 4:
        length = min(lw, n_y - yc)
        if length < max(opts.min_lw // 2, 8):
            break
        window = y[yc : yc + length]
        pt, f, qinfo = exmax(
            window,
            slv=opts.slv,
            f0=f0,
            q0=q0[:, : window.shape[1]] if q0.shape[1] == n_c else None,
            n_iter=opts.n_iter,
            tol=opts.tol,
            weed=opts.weed,
            nstd=opts.nstd,
            cf=opts.cf,
        )
        key = qinfo["key"]
        if key.size > opts.s0 - 1 and pt.size:
            p[yc : yc + length, key] = pt
            f_hat[key] = f
            f0[key] = np.clip(f, opts.fw[0], opts.fw[1])
        yc += step
    return p, f_hat


def pick_from_outlier_probability(
    p: np.ndarray,
    f: np.ndarray,
    *,
    sw: int,
    opts: StaLtaOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """First-event P-picks from outlier-probability columns (independent-channel ``sop``)."""
    p = _as_column_matrix(p)
    n_p, n_c = p.shape
    picks = np.full(n_c, np.nan, dtype=np.float64)
    quality = np.zeros(n_c, dtype=np.float64)
    if n_p <= sw or n_c == 0:
        return picks, quality

    f = np.broadcast_to(np.asarray(f, dtype=np.float64).reshape(-1), (n_c,)).copy()
    thr = (opts.th * f) if opts.th >= 1.0 else np.full(n_c, opts.th)
    thr = np.minimum(thr, opts.tmax)
    pthr = (opts.p_th * f) if opts.p_th >= 1.0 else np.full(n_c, opts.p_th)
    pthr = np.minimum(pthr, opts.tmax)

    sta = uniform_filter1d(p, size=int(sw), axis=0, mode="nearest")
    r = sta - thr.reshape(1, -1)
    sw2 = max(int(round(sw / 2.0)), 1)

    for c in range(n_c):
        above = np.flatnonzero(r[:, c] > 0)
        if above.size == 0:
            continue
        k0 = int(above[0])
        after = np.flatnonzero(r[k0:, c] <= 0)
        k1 = k0 + int(after[0]) if after.size else n_p - 1
        if opts.onset_pick and (k1 - k0) >= 8:
            lo = max(0, k0 - sw2)
            hi = min(n_p, k1 + sw2)
            if hi - lo >= 8:
                tp, oq, _ = es94_autopick(r[lo:hi, c : c + 1] - (pthr[c] - thr[c]))
                oq_c = float(oq[0] / max(1.0 - pthr[c], _EPS))
                if tp[0] > 0 and oq_c >= opts.qth:
                    idx = lo + float(tp[0])
                    if 0 <= idx < n_p:
                        picks[c] = idx
                        quality[c] = oq_c
                        continue
        picks[c] = float(min(max(k0 + sw2, 0), n_p - 1))
        quality[c] = float(max(np.max(r[k0 : k1 + 1, c]) / max(1.0 - thr[c], _EPS), 0.0))
    return picks, quality


def pick_first_breaks(
    samples: np.ndarray,
    sample_rate_hz: float,
    opts: StaLtaOptions | None = None,
    *,
    return_quality: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Pick first breaks on a gather.

    Shot gathers have one arrival per trace, so this runs a **single** two-state
    EM on the full record (pre-break = null, post-break = outlier) and then the
    STA-LTA-OS short-window onset picker. Sliding-window ``hmmscan`` remains
    available for multi-event borehole records.

    Parameters
    ----------
    samples
        Array shaped ``(n_traces, n_samples)``.
    sample_rate_hz
        Sampling frequency in Hz.
    opts
        Algorithm options. ``None`` uses paper defaults scaled to *sample_rate_hz*.

    Returns
    -------
    sample_index
        Float array ``(n_traces,)``. ``NaN`` where no pick was made (dead traces
        or HMM/STA never crossed threshold).
    """
    opts = opts or StaLtaOptions()
    traces = np.asarray(samples, dtype=np.float64)
    if traces.ndim != 2:
        raise ValueError(f"samples must be (n_traces, n_samples), got {traces.shape}")
    n_tr, n_samp = traces.shape
    idx = np.full(n_tr, np.nan, dtype=np.float64)
    qual = np.zeros(n_tr, dtype=np.float64)
    if n_tr == 0 or n_samp < 8:
        return (idx, qual) if return_quality else idx

    dead = np.max(np.abs(traces), axis=1) <= _DEAD_TRACE_EPS
    live = np.flatnonzero(~dead)
    if live.size == 0:
        return (idx, qual) if return_quality else idx

    x = traces[live].T.copy()
    peak = np.max(np.abs(x), axis=0, keepdims=True)
    peak = np.maximum(peak, _EPS)
    x = x / peak
    y = genprep(x, [opts.cf])
    pt, f, qinfo = exmax(
        y,
        slv=opts.slv,
        f0=opts.f0,
        n_iter=opts.n_iter,
        tol=opts.tol,
        weed=opts.weed,
        nstd=opts.nstd,
        cf=opts.cf,
    )
    p = np.zeros_like(y)
    f_hat = np.full(live.size, opts.f0, dtype=np.float64)
    key = qinfo["key"]
    if key.size:
        p[:, key] = pt
        f_hat[key] = f
    _, sw = _window_samples(y.shape[0], sample_rate_hz, opts)
    picks, q = pick_from_outlier_probability(p, f_hat, sw=sw, opts=opts)
    finite = np.isfinite(picks)
    picks[finite] = np.clip(picks[finite], 0, n_samp - 1)
    idx[live] = picks
    qual[live] = q
    return (idx, qual) if return_quality else idx


def pick_first_breaks_ms(
    samples: np.ndarray,
    sample_rate_ms: float,
    opts: StaLtaOptions | None = None,
) -> np.ndarray:
    """Like :func:`pick_first_breaks` but returns times in milliseconds (NaN = no pick)."""
    dt_ms = float(sample_rate_ms)
    fs_hz = 1000.0 / max(dt_ms, _EPS)
    idx = pick_first_breaks(samples, fs_hz, opts)
    fb_ms = idx * dt_ms
    fb_ms[~np.isfinite(idx)] = np.nan
    return fb_ms


def pick_first_breaks_ms_from_shot_gather(
    gather: ShotGather,
    opts: StaLtaOptions | None = None,
) -> np.ndarray:
    """Pick first-break times (ms) on a native :class:`ShotGather`."""
    dt_ms = float(gather.sample_rate_us) / 1000.0
    return pick_first_breaks_ms(np.asarray(gather.traces), dt_ms, opts)


def hmmscan(
    x: np.ndarray,
    fs_hz: float = 4000.0,
    opts: StaLtaOptions | None = None,
) -> dict[str, Any]:
    """Scan column-vector traces for events (MATLAB ``hmmscan``-style output).

    Returns a dict with ``s`` / ``e`` (seconds), optional ``p`` (P-picks, seconds),
    ``rs`` / ``rb`` detection statistics, and ``pt`` outlier probabilities.
    """
    opts = opts or StaLtaOptions()
    x = _as_column_matrix(x)
    peak = np.max(np.abs(x))
    if peak <= _EPS:
        return {"pt": np.zeros_like(x), "rs": np.zeros_like(x), "rb": np.zeros((x.shape[0],))}
    y = genprep(x / peak, [opts.cf])
    p, f_hat = outlier_probability_scan(y, fs_hz=fs_hz, opts=opts)
    _, sw = _window_samples(y.shape[0], fs_hz, opts)
    picks, quality = pick_from_outlier_probability(p, f_hat, sw=sw, opts=opts)
    sta = uniform_filter1d(p, size=sw, axis=0, mode="nearest")
    thr = np.minimum(opts.th * f_hat, opts.tmax)
    rs = sta - thr.reshape(1, -1)
    rb = np.mean(rs, axis=1)
    starts = []
    ends = []
    p_times = []
    for c in range(p.shape[1]):
        if not np.isfinite(picks[c]):
            continue
        starts.append(float(picks[c] / fs_hz))
        ends.append(float(min(picks[c] + sw, p.shape[0] - 1) / fs_hz))
        p_times.append(float(picks[c] / fs_hz))
    return {
        "s": np.asarray(starts, dtype=np.float64),
        "e": np.asarray(ends, dtype=np.float64),
        "p": np.asarray(p_times, dtype=np.float64),
        "pq": quality,
        "pt": p,
        "rs": rs,
        "rb": rb,
        "f": f_hat,
    }
