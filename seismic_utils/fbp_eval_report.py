"""Write FBP validation reports: scalars, Plotly stats HTML, gather PNGs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .dataset import ShotGather

HIT_BUFFERS_PX = (1, 3, 5, 7, 9)


def origin_name_map(origin_id_map: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in origin_id_map.items()}


def annotate_trace_frame(
    df: pd.DataFrame,
    *,
    origin_id_map: Dict[str, int],
    sample_rate_ms_by_origin: Dict[str, float],
    default_sample_rate_ms: float = 2.0,
) -> pd.DataFrame:
    """Add Origin, |error|, and millisecond columns."""
    out = df.copy()
    for col in ("OriginId", "GatherId", "ShotId", "ReceiverId", "Predictions"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    for col in ("Offset", "Errors", "Probabilities", "GatherCoverage", "ExpectedCoverage"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    id_to_name = origin_name_map(origin_id_map)
    if "OriginId" in out.columns:
        out["Origin"] = out["OriginId"].map(id_to_name)
    else:
        out["Origin"] = "unknown"
    abs_err = out["Errors"].abs() if "Errors" in out.columns else pd.Series(np.nan, index=out.index)
    out["AbsError"] = abs_err

    def _rate(origin: Any) -> float:
        name = str(origin)
        if name in sample_rate_ms_by_origin:
            return float(sample_rate_ms_by_origin[name])
        name_l = name.lower()
        for key, value in sample_rate_ms_by_origin.items():
            key_l = str(key).lower()
            if key_l in name_l or name_l in key_l:
                return float(value)
        return float(default_sample_rate_ms)

    dt = out["Origin"].map(_rate)
    out["SampleRateMs"] = dt
    out["ErrorMs"] = out["Errors"] * dt
    out["AbsErrorMs"] = abs_err * dt
    return out

    def _rate(origin: Any) -> float:
        name = str(origin)
        if name in sample_rate_ms_by_origin:
            return float(sample_rate_ms_by_origin[name])
        name_l = name.lower()
        for key, value in sample_rate_ms_by_origin.items():
            key_l = str(key).lower()
            if key_l in name_l or name_l in key_l:
                return float(value)
        return float(default_sample_rate_ms)

    dt = out["Origin"].map(_rate)
    out["SampleRateMs"] = dt
    out["ErrorMs"] = out["Errors"] * dt
    out["AbsErrorMs"] = abs_err * dt
    return out


def headline_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Scalar metrics from the per-trace evaluator table."""
    labeled = df["Errors"].notna() if "Errors" in df.columns else pd.Series(False, index=df.index)
    n_traces = int(len(df))
    n_labeled = int(labeled.sum())
    abs_err = df.loc[labeled, "AbsError"] if n_labeled else pd.Series(dtype=float)
    err = df.loc[labeled, "Errors"] if n_labeled else pd.Series(dtype=float)
    abs_ms = df.loc[labeled, "AbsErrorMs"] if n_labeled else pd.Series(dtype=float)
    err_ms = df.loc[labeled, "ErrorMs"] if n_labeled else pd.Series(dtype=float)

    out: Dict[str, Any] = {
        "n_traces": n_traces,
        "n_labeled": n_labeled,
        "n_unlabeled": n_traces - n_labeled,
        "n_gathers": int(df.groupby(["OriginId", "GatherId", "ShotId"], dropna=False).ngroups)
        if n_traces
        else 0,
    }
    for buf in HIT_BUFFERS_PX:
        key = f"HitRate{buf}px"
        out[key] = float((abs_err < buf).mean()) if n_labeled else None
    out["MeanAbsoluteError"] = float(abs_err.mean()) if n_labeled else None
    out["MedianAbsoluteError"] = float(abs_err.median()) if n_labeled else None
    out["P90AbsoluteError"] = float(abs_err.quantile(0.90)) if n_labeled else None
    out["P95AbsoluteError"] = float(abs_err.quantile(0.95)) if n_labeled else None
    out["RootMeanSquaredError"] = float(np.sqrt((abs_err ** 2).mean())) if n_labeled else None
    out["MeanBiasError"] = float(err.mean()) if n_labeled else None
    out["MeanAbsoluteErrorMs"] = float(abs_ms.mean()) if n_labeled else None
    out["MedianAbsoluteErrorMs"] = float(abs_ms.median()) if n_labeled else None
    out["MeanBiasErrorMs"] = float(err_ms.mean()) if n_labeled else None
    if "GatherCoverage" in df.columns and "ExpectedCoverage" in df.columns:
        expected = float(df["ExpectedCoverage"].sum())
        out["GatherCoverage"] = float(df["GatherCoverage"].sum() / expected) if expected else None
    elif "GatherCoverage" in df.columns and n_traces:
        out["GatherCoverage"] = float(df["GatherCoverage"].mean())
    else:
        out["GatherCoverage"] = None
    return out


def offset_bin_table(df: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    labeled = df[df["Errors"].notna()].copy()
    if labeled.empty or "Offset" not in labeled.columns:
        return pd.DataFrame()
    labeled["Offset"] = pd.to_numeric(labeled["Offset"], errors="coerce")
    labeled = labeled[np.isfinite(labeled["Offset"].to_numpy(dtype=np.float64, copy=False))]
    if labeled.empty or labeled["Offset"].nunique() < 2:
        return pd.DataFrame()
    try:
        labeled["offset_bin"] = pd.qcut(labeled["Offset"], q=min(n_bins, labeled["Offset"].nunique()), duplicates="drop")
    except ValueError:
        labeled["offset_bin"] = pd.cut(labeled["Offset"], bins=min(n_bins, max(labeled["Offset"].nunique(), 2)))
    rows = []
    for interval, g in labeled.groupby("offset_bin", observed=True):
        abs_err = g["AbsError"]
        left = float(interval.left) if hasattr(interval, "left") else float("nan")
        right = float(interval.right) if hasattr(interval, "right") else float("nan")
        rows.append(
            {
                "offset_left": left,
                "offset_right": right,
                "offset_mid": 0.5 * (left + right),
                "n": int(len(g)),
                "HitRate1px": float((abs_err < 1).mean()),
                "HitRate5px": float((abs_err < 5).mean()),
                "MAE": float(abs_err.mean()),
                "MBE": float(g["Errors"].mean()),
            }
        )
    return pd.DataFrame(rows)


def gather_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per shot-line gather."""
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["OriginId", "GatherId", "ShotId"], dropna=False, sort=False)
    agg: Dict[str, Any] = {
        "Origin": ("Origin", "first"),
        "n_traces": ("Errors", "size"),
        "n_labeled": ("Errors", "count"),
        "MAE": ("AbsError", "mean"),
        "P90AbsError": (
            "AbsError",
            lambda s: float(s.quantile(0.90)) if s.notna().any() else np.nan,
        ),
        "MaxAbsError": ("AbsError", "max"),
        "HitRate1px": (
            "AbsError",
            lambda s: float((s.dropna() < 1).mean()) if s.notna().any() else np.nan,
        ),
        "MBE": ("Errors", "mean"),
    }
    if "GatherCoverage" in df.columns:
        agg["Coverage"] = ("GatherCoverage", "mean")
    return grouped.agg(**agg).reset_index()


def pick_gallery_gathers(
    gather_df: pd.DataFrame,
    *,
    n_worst: int = 8,
    n_typical: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = gather_df.dropna(subset=["MAE"]).copy()
    if ranked.empty:
        return ranked, ranked
    worst = ranked.sort_values(["P90AbsError", "MAE"], ascending=False).head(int(n_worst))
    remaining = ranked.drop(index=worst.index, errors="ignore")
    if remaining.empty:
        remaining = ranked
    remaining = remaining.sort_values("MAE", ascending=True)
    mid = len(remaining) // 2
    half = int(n_typical) // 2
    start = max(mid - half, 0)
    typical = remaining.iloc[start : start + int(n_typical)]
    if typical.empty:
        typical = remaining.head(int(n_typical))
    return worst, typical


def write_trace_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    try:
        parquet = path.with_suffix(".parquet")
        df.to_parquet(parquet, index=False)
        return parquet
    except Exception:
        csv_path = path.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        return csv_path


def write_stats_html(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    offset_df: pd.DataFrame,
    gather_df: pd.DataFrame,
    path: Path,
) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labeled = df[df["Errors"].notna()]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Signed pick error (samples)",
            "CDF of |error| (samples)",
            "HR@1px vs offset",
            "MAE vs offset",
        ),
    )
    if not labeled.empty:
        errors = labeled["Errors"].clip(-80, 80)
        fig.add_trace(
            go.Histogram(x=errors, nbinsx=80, name="error", marker_color="#1f77b4"),
            row=1,
            col=1,
        )
        abs_all = np.sort(labeled["AbsError"].to_numpy())
        n_abs = len(abs_all)
        if n_abs > 4000:
            idx = np.linspace(0, n_abs - 1, 4000).astype(int)
            abs_sorted = abs_all[idx]
            cdf = (idx + 1) / n_abs
        else:
            abs_sorted = abs_all
            cdf = np.arange(1, n_abs + 1) / max(n_abs, 1)
        fig.add_trace(
            go.Scatter(x=abs_sorted, y=cdf, mode="lines", name="CDF", line=dict(color="#d62728")),
            row=1,
            col=2,
        )
    if not offset_df.empty:
        fig.add_trace(
            go.Scatter(
                x=offset_df["offset_mid"],
                y=offset_df["HitRate1px"],
                mode="lines+markers",
                name="HR@1px",
                line=dict(color="#2ca02c"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=offset_df["offset_mid"],
                y=offset_df["MAE"],
                mode="lines+markers",
                name="MAE",
                line=dict(color="#ff7f0e"),
            ),
            row=2,
            col=2,
        )
    fig.update_xaxes(title_text="error (samples)", row=1, col=1)
    fig.update_xaxes(title_text="|error| (samples)", row=1, col=2)
    fig.update_xaxes(title_text="offset", row=2, col=1)
    fig.update_xaxes(title_text="offset", row=2, col=2)
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_yaxes(title_text="fraction", row=1, col=2, range=[0, 1])
    fig.update_yaxes(title_text="HR@1px", row=2, col=1, range=[0, 1])
    fig.update_yaxes(title_text="MAE (samples)", row=2, col=2)
    fig.update_layout(
        title="FBP validation statistics",
        showlegend=False,
        height=740,
        margin=dict(t=60, l=50, r=20, b=50),
    )

    headline = (
        f"n_gathers={metrics.get('n_gathers')}  n_labeled={metrics.get('n_labeled')}  "
        f"HR@1={_fmt(metrics.get('HitRate1px'))}  "
        f"MAE={_fmt(metrics.get('MeanAbsoluteError'))} samples "
        f"({_fmt(metrics.get('MeanAbsoluteErrorMs'))} ms)  "
        f"coverage={_fmt(metrics.get('GatherCoverage'))}"
    )
    table_html = ""
    if not gather_df.empty:
        preview = gather_df.sort_values("MAE", ascending=False).head(25)
        cols = [
            c
            for c in (
                "Origin",
                "GatherId",
                "ShotId",
                "n_labeled",
                "MAE",
                "P90AbsError",
                "HitRate1px",
            )
            if c in preview.columns
        ]
        table_html = preview[cols].to_html(index=False, float_format=lambda x: f"{x:.4g}")

    path = Path(path)
    path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>FBP eval stats</title></head><body>"
        f"<h1>Validation statistics</h1><p>{headline}</p>"
        f"{fig.to_html(full_html=False, include_plotlyjs='cdn')}"
        "<h2>Worst gathers (by MAE, top 25)</h2>"
        f"{table_html or '<p>No labeled gathers.</p>'}"
        "</body></html>\n",
        encoding="utf-8",
    )
    return path


def plot_gather_residual(
    gather: ShotGather,
    pred_ms: np.ndarray,
    out_path: Path,
    *,
    subtitle: str = "",
) -> Path:
    pred = np.asarray(pred_ms, dtype=np.float64).reshape(-1)
    if pred.shape[0] != gather.n_traces:
        raise ValueError(f"pred length {pred.shape[0]} != n_traces {gather.n_traces}")
    residual = pred - gather.first_breaks_ms
    amp = gather.traces.T
    limit = float(np.percentile(np.abs(amp), 99.0) or 1.0)
    time_ms = gather.time_ms
    extent = (0, max(gather.n_traces - 1, 1), time_ms[-1] if len(time_ms) else 1.0, time_ms[0] if len(time_ms) else 0.0)
    x = np.arange(gather.n_traces, dtype=np.float64)

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(11, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.1]},
    )
    ax.imshow(
        amp,
        aspect="auto",
        cmap="gray",
        vmin=-limit,
        vmax=limit,
        extent=extent,
        interpolation="nearest",
    )
    if np.any(gather.labeled_mask):
        ax.plot(x, gather.first_breaks_ms, color="yellow", lw=1.4, label="Reference")
    if np.any(np.isfinite(pred)):
        ax.plot(x, pred, color="lime", lw=1.8, label="Prediction")
    ax.set_ylabel("Time (ms)")
    ax.set_title(subtitle or f"shot={gather.shot_id} line={gather.line_id}")
    ax.legend(loc="upper right", fontsize=8)

    axr.axhline(0.0, color="black", lw=0.7)
    axr.plot(x, residual, color="tab:red", lw=0.9)
    axr.set_xlabel("Trace index")
    axr.set_ylabel("Residual (ms)")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(out_path)


def write_worst_html(
    rows: Sequence[Dict[str, Any]],
    path: Path,
    *,
    title: str,
) -> Path:
    cards = []
    for row in rows:
        rel = row["image"]
        cards.append(
            "<div class='card'>"
            f"<h3>{row.get('label', '')}</h3>"
            f"<p>{row.get('stats', '')}</p>"
            f"<img src='{rel}' alt='{row.get('label', '')}'/>"
            "</div>"
        )
    path = Path(path)
    path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>"
        f"{title}</title>"
        "<style>"
        "body{font-family:sans-serif;margin:1.5rem;background:#111;color:#eee}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem}"
        ".card{background:#1b1b1b;padding:0.8rem;border-radius:8px}"
        "img{width:100%;height:auto;background:#000}"
        "a{color:#8ab4f8}"
        "</style></head><body>"
        f"<p><a href='index.html'>Back to index</a></p><h1>{title}</h1>"
        f"<div class='grid'>{''.join(cards)}</div>"
        "</body></html>\n",
        encoding="utf-8",
    )
    return path


def write_index_html(path: Path, *, title: str, metrics: Dict[str, Any], links: Dict[str, str]) -> Path:
    rows = "".join(
        f"<tr><th>{k}</th><td>{_fmt(v)}</td></tr>"
        for k, v in metrics.items()
        if k
        in {
            "n_gathers",
            "n_traces",
            "n_labeled",
            "HitRate1px",
            "HitRate3px",
            "HitRate5px",
            "MeanAbsoluteError",
            "MeanAbsoluteErrorMs",
            "MedianAbsoluteError",
            "P90AbsoluteError",
            "MeanBiasError",
            "GatherCoverage",
            "loss",
        }
    )
    link_items = "".join(f"<li><a href='{href}'>{name}</a></li>" for name, href in links.items())
    path = Path(path)
    path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title>"
        "<style>body{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:0.35rem 0.6rem;text-align:left}"
        "</style></head><body>"
        f"<h1>{title}</h1>"
        f"<h2>Headline metrics</h2><table>{rows}</table>"
        f"<h2>Artifacts</h2><ul>{link_items}</ul>"
        "</body></html>\n",
        encoding="utf-8",
    )
    return path


def write_report_md(
    path: Path,
    *,
    meta: Dict[str, Any],
    metrics: Dict[str, Any],
    offset_df: pd.DataFrame,
    worst_names: Iterable[str],
    typical_names: Iterable[str],
) -> Path:
    lines = [
        f"# FBP validation — {meta.get('run_name', '')}",
        "",
        f"- **Picker:** {meta.get('picker', 'fbpunet')}",
        f"- **Checkpoint:** `{meta.get('checkpoint', '')}`",
        f"- **Config:** `{meta.get('model_config', '')}`",
        f"- **Encoder:** {meta.get('encoder', '')}",
        f"- **Sites:** {', '.join(meta.get('sites') or [])}",
        f"- **Fold:** {meta.get('fold') or '—'}",
        f"- **Backend:** {meta.get('backend', '')}",
        f"- **Gathers:** {metrics.get('n_gathers')}  |  traces: {metrics.get('n_traces')}  "
        f"| labeled: {metrics.get('n_labeled')}",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key in (
        "HitRate1px",
        "HitRate3px",
        "HitRate5px",
        "HitRate7px",
        "HitRate9px",
        "MeanAbsoluteError",
        "MedianAbsoluteError",
        "P90AbsoluteError",
        "RootMeanSquaredError",
        "MeanBiasError",
        "MeanAbsoluteErrorMs",
        "MeanBiasErrorMs",
        "GatherCoverage",
        "loss",
    ):
        if key in metrics:
            lines.append(f"| {key} | {_fmt(metrics[key])} |")
    lines.extend(["", "## Offset bins", ""])
    if offset_df.empty:
        lines.append("_Not enough offset variation to bin._")
    else:
        lines.append("| offset mid | n | HR@1 | HR@5 | MAE | MBE |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in offset_df.iterrows():
            lines.append(
                f"| {row['offset_mid']:.1f} | {int(row['n'])} | "
                f"{row['HitRate1px']:.4f} | {row['HitRate5px']:.4f} | "
                f"{row['MAE']:.3g} | {row['MBE']:.3g} |"
            )
    lines.extend(["", "## Worst gathers", ""])
    for name in worst_names:
        lines.append(f"- `{name}`")
    lines.extend(["", "## Typical gathers", ""])
    for name in typical_names:
        lines.append(f"- `{name}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- [index.html](index.html)",
            "- [stats.html](stats.html) (Plotly)",
            "- [worst.html](worst.html)",
            "- [typical.html](typical.html)",
            "",
        ]
    )
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)
