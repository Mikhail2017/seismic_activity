"""Plotting helpers for seismic shot gathers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .dataset import ShotGather

BEFORE_COLOR = (0.15, 0.35, 0.95)  # blue
AFTER_COLOR = (0.90, 0.15, 0.15)  # red
UNLABELED_COLOR = (0.55, 0.55, 0.55)  # gray
OVERLAY_ALPHA = 0.28


PRED_COLOR = "lime"
PICK_DISPLAY_MODES = ("reference", "prediction", "both")


def _region_masks_from_fb(
    gather: ShotGather,
    first_breaks_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Boolean masks shaped ``(n_samples, n_traces)``.

    Returns ``(before, after, unlabeled)``. Unlabeled covers every sample on
    traces without a finite first-break pick in *first_breaks_ms*.
    """
    fb = np.asarray(first_breaks_ms, dtype=np.float64).reshape(-1)
    if fb.shape[0] != gather.n_traces:
        raise ValueError(
            f"first_breaks_ms length {fb.shape[0]} != n_traces {gather.n_traces}"
        )
    labeled = np.isfinite(fb)
    time = gather.time_ms[:, None]  # (samples, 1)
    fb2 = fb[None, :]
    labeled2 = labeled[None, :]
    before = labeled2 & (time < fb2)
    after = labeled2 & (time >= fb2)
    unlabeled = np.broadcast_to(~labeled[None, :], before.shape).copy()
    return before, after, unlabeled


def _region_masks(gather: ShotGather) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Region masks from ground-truth first breaks."""
    return _region_masks_from_fb(gather, gather.first_breaks_ms)


def plot_shot_gather(
    gather: ShotGather,
    *,
    show_first_breaks: bool = True,
    highlight_regions: bool = False,
    predicted_first_breaks_ms: np.ndarray | None = None,
    pick_display: str = "reference",
    clip_percentile: float = 99.0,
    figsize: tuple[float, float] = (10, 8),
) -> Figure:
    """
    Plot a 2D seismic image for one SHOTID.

    X-axis: trace index (ordered along receivers).
    Y-axis: time in milliseconds.

    When *highlight_regions* is True, samples before the first break are tinted
    blue, after are tinted red, unlabeled traces are tinted gray, and a
    class-count histogram is shown below the gather.

    *pick_display* controls which pick curves are drawn (and which drive the
    region tint): ``\"reference\"``, ``\"prediction\"``, or ``\"both\"``.
    """
    mode = pick_display.strip().lower()
    if mode not in PICK_DISPLAY_MODES:
        raise ValueError(f"pick_display must be one of {PICK_DISPLAY_MODES}, got {pick_display!r}")

    pred_ms = None
    if predicted_first_breaks_ms is not None:
        pred_ms = np.asarray(predicted_first_breaks_ms, dtype=np.float64).reshape(-1)
        if pred_ms.shape[0] != gather.n_traces:
            raise ValueError(
                f"predicted_first_breaks_ms length {pred_ms.shape[0]} != n_traces {gather.n_traces}"
            )

    if mode in {"prediction", "both"} and pred_ms is None:
        raise ValueError(f"pick_display={mode!r} requires predicted_first_breaks_ms")

    # Region tint follows the active pick source (GT when showing both).
    if mode == "prediction":
        region_fb = pred_ms
        hist_title = "Class balance (prediction)"
    else:
        region_fb = gather.first_breaks_ms
        hist_title = "Class balance (reference)"

    amp = gather.traces.T  # (samples, traces) for imshow with time vertical
    limit = float(np.percentile(np.abs(amp), clip_percentile))
    if limit <= 0:
        limit = 1.0

    time_ms = gather.time_ms
    extent = (0, gather.n_traces - 1, time_ms[-1], time_ms[0])

    if highlight_regions:
        fig, (ax, ax_hist) = plt.subplots(
            2,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [3.2, 1.0]},
            layout="constrained",
        )
    else:
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 0.75), layout="constrained")
        ax_hist = None

    ax.imshow(
        amp,
        aspect="auto",
        cmap="gray",
        vmin=-limit,
        vmax=limit,
        extent=extent,
        interpolation="nearest",
    )

    before_count = after_count = unlabeled_count = 0
    legend_handles: list = []
    if highlight_regions:
        before, after, unlabeled = _region_masks_from_fb(gather, region_fb)
        before_count = int(before.sum())
        after_count = int(after.sum())
        unlabeled_count = int(unlabeled.sum())

        overlay = np.zeros((gather.n_samples, gather.n_traces, 4), dtype=np.float32)
        overlay[before] = (*BEFORE_COLOR, OVERLAY_ALPHA)
        overlay[after] = (*AFTER_COLOR, OVERLAY_ALPHA)
        overlay[unlabeled] = (*UNLABELED_COLOR, OVERLAY_ALPHA)
        ax.imshow(overlay, aspect="auto", extent=extent, interpolation="nearest")

        legend_handles = [
            Patch(facecolor=BEFORE_COLOR, alpha=0.55, label="Before first break"),
            Patch(facecolor=AFTER_COLOR, alpha=0.55, label="After first break"),
            Patch(facecolor=UNLABELED_COLOR, alpha=0.55, label="Unlabeled"),
        ]

    ax.set_xlabel("Trace index (CHANNEL order)")
    ax.set_ylabel("Time (ms)")
    if gather.line_id is not None:
        title = (
            f"Gather {gather.gather_id}  |  shot={gather.shot_id}  "
            f"line={gather.line_id}  ({gather.n_traces} traces)"
        )
    else:
        title = f"SHOTID {gather.shot_id}  ({gather.n_traces} traces)"
    if mode != "reference":
        title = f"{title}  |  picks={mode}"
    ax.set_title(title)

    x = np.arange(gather.n_traces, dtype=np.float64)
    if show_first_breaks and mode in {"reference", "both"} and np.any(gather.labeled_mask):
        y = gather.first_breaks_ms.copy()
        line_color = "yellow" if highlight_regions else "red"
        (line,) = ax.plot(x, y, color=line_color, linewidth=1.5, label="Reference FB")
        legend_handles.append(line)

    if show_first_breaks and mode in {"prediction", "both"} and pred_ms is not None:
        y_pred = pred_ms.copy()
        if np.any(np.isfinite(y_pred)):
            (pline,) = ax.plot(
                x, y_pred, color=PRED_COLOR, linewidth=2.0, label="Predicted FB"
            )
            legend_handles.append(pline)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right")

    if ax_hist is not None:
        labels = ["Before", "After", "Unlabeled"]
        counts = [before_count, after_count, unlabeled_count]
        colors = [BEFORE_COLOR, AFTER_COLOR, UNLABELED_COLOR]
        bars = ax_hist.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.6)
        ax_hist.set_ylabel("Sample count")
        ax_hist.set_title(hist_title)
        total = before_count + after_count + unlabeled_count
        for bar, count in zip(bars, counts):
            pct = 100.0 * count / total if total else 0.0
            ax_hist.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax_hist.set_ylim(0, max(counts) * 1.25 if max(counts) > 0 else 1.0)

    return fig
