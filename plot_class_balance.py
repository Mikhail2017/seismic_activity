"""Plot global before / after / unlabeled sample counts for all HDF5 assets."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read all HDF5 files in a directory and plot total sample counts "
            "before / after first break and unlabeled (same definition as the Gradio viewer)."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory with HDF5 assets (default: /home/mika/data/seismic_activity)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to save the figure (PNG/PDF)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window (useful with --output)",
    )
    args = parser.parse_args(argv)

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm

    from seismic_utils.class_balance import ClassCounts, count_classes_in_hdf5
    from seismic_utils.dataset import DEFAULT_DATA_DIR, list_dataset_files
    from seismic_utils.plotting import AFTER_COLOR, BEFORE_COLOR, UNLABELED_COLOR

    data_dir = Path(args.data_dir or DEFAULT_DATA_DIR).expanduser()
    files = list_dataset_files(data_dir)
    if not files:
        raise SystemExit(f"No HDF5 files found in {data_dir}")

    counts: list[ClassCounts] = []
    for path in tqdm(files, desc="Scanning HDF5", unit="file"):
        result = count_classes_in_hdf5(path)
        counts.append(result)
        print(
            f"{result.asset}: before={result.before:,} after={result.after:,} "
            f"unlabeled={result.unlabeled:,} traces={result.n_traces:,}"
        )

    total_before = sum(c.before for c in counts)
    total_after = sum(c.after for c in counts)
    total_unlabeled = sum(c.unlabeled for c in counts)
    print(
        f"TOTAL: before={total_before:,} after={total_after:,} "
        f"unlabeled={total_unlabeled:,}"
    )

    labels = ["Before", "After", "Unlabeled"]
    colors = [BEFORE_COLOR, AFTER_COLOR, UNLABELED_COLOR]
    totals = [total_before, total_after, total_unlabeled]
    grand = sum(totals)

    assets = [c.asset for c in counts]
    before_vals = np.array([c.before for c in counts], dtype=np.float64)
    after_vals = np.array([c.after for c in counts], dtype=np.float64)
    unlabeled_vals = np.array([c.unlabeled for c in counts], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")

    ax0 = axes[0]
    bars = ax0.bar(labels, totals, color=colors, edgecolor="black", linewidth=0.6)
    ax0.set_ylabel("Sample count")
    ax0.set_title(f"Total class balance ({len(counts)} assets)")
    for bar, value in zip(bars, totals):
        pct = 100.0 * value / grand if grand else 0.0
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax0.set_ylim(0, max(totals) * 1.25 if max(totals) > 0 else 1.0)

    ax1 = axes[1]
    x = np.arange(len(assets))
    width = 0.25
    ax1.bar(x - width, before_vals, width, label="Before", color=BEFORE_COLOR, edgecolor="black", linewidth=0.4)
    ax1.bar(x, after_vals, width, label="After", color=AFTER_COLOR, edgecolor="black", linewidth=0.4)
    ax1.bar(x + width, unlabeled_vals, width, label="Unlabeled", color=UNLABELED_COLOR, edgecolor="black", linewidth=0.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(assets, rotation=20, ha="right")
    ax1.set_ylabel("Sample count")
    ax1.set_title("Per-asset class balance")
    ax1.legend(loc="upper right")

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Saved figure to {output_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
