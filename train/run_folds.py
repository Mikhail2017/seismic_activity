#!/usr/bin/env python3
"""Train + lateral-clean eval for leave-one-site-out folds, then write a summary.

Default is folds A–D from TRAINING.md (3 train sites / 1 valid site). Each fold
runs ``fbp_train.py`` then ``fbp_eval.py --lateral-clean``. A living summary
under ``report/folds_AD_<stamp>/report.md`` links the per-fold train and eval
reports and compares headline metrics.

Example::

    python train/run_folds.py --data-dir /tmp/data
    python train/run_folds.py --folds A,B --data-dir /tmp/data --dry-run
    python train/run_folds.py --eval-only --data-dir /tmp/data
    python train/run_folds.py --eval-only --folds C --sweep report/folds_AD_…/sweep.json \\
        --data-dir /tmp/data --before-after-decoder change_point
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from seismic_utils.dataset import DEFAULT_DATA_DIR
from seismic_utils.fb_smooth import BEFORE_AFTER_DECODERS
from seismic_utils.pick_clean import (
    DEFAULT_LATERAL_MAX_DEV,
    DEFAULT_LATERAL_MAX_FLAG_FRAC,
    DEFAULT_LATERAL_MIN_ANCHORS,
    DEFAULT_LATERAL_WINDOW,
)
from seismic_utils.pickers import PICKER_BEFORE_AFTER, PICKER_FBPUNET

# Leave-one-site-out on the four local surveys (TRAINING.md / fbp_train.SITE_FOLDS).
FOLDS_AD: dict[str, dict[str, list[str]]] = {
    "A": {"train": ["Lalor", "Brunswick", "Sudbury"], "valid": ["Halfmile"]},
    "B": {"train": ["Lalor", "Brunswick", "Halfmile"], "valid": ["Sudbury"]},
    "C": {"train": ["Halfmile", "Lalor", "Sudbury"], "valid": ["Brunswick"]},
    "D": {"train": ["Sudbury", "Halfmile", "Brunswick"], "valid": ["Lalor"]},
}

DEFAULT_FOLDS = ("A", "B", "C", "D")
_HEADLINE = (
    "HitRate1px",
    "HitRate5px",
    "MeanAbsoluteError",
    "RootMeanSquaredError",
    "P90AbsoluteError",
    "MeanBiasError",
    "GatherCoverage",
)


def _fold_letter(token: str) -> str:
    key = token.strip().upper().replace("_", "").replace("-", "").replace(" ", "")
    if key.startswith("FOLD"):
        key = key[4:]
    return key


def parse_folds(spec: str | None, *, allowed: Mapping[str, Any] | None = None) -> list[str]:
    """Parse ``A-D``, ``A,C``, or ``A B D`` into unique fold letters."""
    known = dict(allowed or FOLDS_AD)
    raw = (spec or ",".join(DEFAULT_FOLDS)).strip()
    if not raw:
        raise ValueError("fold spec is empty")
    tokens: list[str] = []
    for chunk in raw.replace(" ", ",").split(","):
        part = chunk.strip()
        if not part:
            continue
        upper = part.upper().replace("_", "")
        if "-" in upper:
            start, _, end = upper.partition("-")
            start_l, end_l = _fold_letter(start), _fold_letter(end)
            if len(start_l) == 1 and len(end_l) == 1 and start_l.isalpha() and end_l.isalpha():
                lo, hi = ord(start_l), ord(end_l)
                if hi < lo:
                    raise ValueError(f"empty fold range: {part}")
                tokens.extend(chr(i) for i in range(lo, hi + 1))
                continue
        tokens.append(_fold_letter(part))
    out: list[str] = []
    for fold in tokens:
        if fold not in known:
            known_s = ", ".join(sorted(known))
            raise ValueError(f"unknown fold {fold!r}; expected one of: {known_s}")
        if fold not in out:
            out.append(fold)
    if not out:
        raise ValueError("no folds selected")
    return out


def parse_ckpt_dirs(values: Sequence[str], folds: Sequence[str]) -> dict[str, Path]:
    """Parse ``A=/path`` or a single path when only one fold is selected."""
    out: dict[str, Path] = {}
    for raw in values:
        text = (raw or "").strip()
        if not text:
            continue
        if "=" in text:
            letter, _, path = text.partition("=")
        elif ":" in text and not text.startswith("/") and not Path(text).exists():
            letter, _, path = text.partition(":")
        else:
            if len(folds) != 1:
                raise ValueError(
                    f"--ckpt-dir {text!r} needs FOLD=DIR when more than one fold is selected"
                )
            out[folds[0]] = Path(text).expanduser()
            continue
        fold = _fold_letter(letter)
        if fold not in FOLDS_AD:
            raise ValueError(f"unknown fold in --ckpt-dir: {letter!r}")
        out[fold] = Path(path).expanduser()
    return out


def find_train_report(experiment_dir: Path | None, report_root: Path) -> Path | None:
    if experiment_dir is None:
        return None
    candidate = Path(report_root) / Path(experiment_dir).name / "report.md"
    return candidate if candidate.is_file() else None


def find_fold_experiment(
    fold: str,
    *,
    output_root: Path,
    explicit: Mapping[str, Path] | None = None,
    sweep: Mapping[str, Any] | None = None,
) -> Path:
    """Resolve a fold's training directory (explicit, sweep.json, or newest output/)."""
    if explicit and fold in explicit:
        path = Path(explicit[fold]).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Fold {fold} --ckpt-dir is not a directory: {path}")
        return path
    if sweep:
        for row in sweep.get("folds") or []:
            if str(row.get("fold")) != fold:
                continue
            raw = row.get("experiment_dir") or row.get("ckpt")
            if not raw:
                break
            path = Path(raw).expanduser()
            if path.is_file():
                path = path.parent
            path = path.resolve()
            if not path.is_dir():
                raise FileNotFoundError(f"Fold {fold} sweep path is missing: {path}")
            return path
    matches = [
        p
        for p in Path(output_root).glob(f"train_fold{fold}_*")
        if p.is_dir()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No output/train_fold{fold}_* under {output_root}. "
            "Pass --ckpt-dir FOLD=DIR or --sweep report/folds_…/sweep.json"
        )
    matches.sort(key=lambda p: p.stat().st_mtime)
    return matches[-1].resolve()


def rel_md_link(from_dir: Path, target: Path | None) -> str | None:
    """Markdown-relative path from *from_dir* to *target*, if it exists."""
    if target is None:
        return None
    path = Path(target).resolve()
    if not path.exists():
        return None
    try:
        return Path(os.path.relpath(path, start=Path(from_dir).resolve())).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_folds_summary(
    report_md: Path,
    *,
    folds: Sequence[str],
    fold_specs: Mapping[str, Mapping[str, Sequence[str]]],
    rows: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
) -> Path:
    """Write the sweep index markdown. Returns *report_md*."""
    report_md = Path(report_md)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    by_fold = {str(r.get("fold")): r for r in rows}
    stamp = meta.get("stamp") or ""
    lines = [
        f"# Fold sweep — {meta.get('label', 'A–D')}",
        "",
        f"**Status:** {meta.get('status', 'running')}",
        f"**Started:** {meta.get('started', '')}",
        f"**Finished:** {meta.get('finished') or '—'}",
        f"**Folds:** {', '.join(folds)}",
        f"**Picker:** {meta.get('picker', '')}",
        f"**GeoNorm:** {meta.get('geonorm', '')}",
        f"**Backend:** {meta.get('backend', '')}",
        f"**Lateral clean:** {'on' if meta.get('lateral_clean') else 'off'}",
        f"**Before/after decoder:** {meta.get('before_after_decoder') or 'checkpoint / legacy'}",
        f"**Data dir:** `{meta.get('data_dir', '')}`",
        "",
        (
            "Leave-one-site-out (TRAINING.md). "
            + (
                "Eval-only: existing best checkpoints, no training."
                if meta.get("eval_only")
                else "Each fold trains, then evaluates the best checkpoint with `--lateral-clean`."
            )
        ),
        "",
        "## Folds",
        "",
        "| Fold | Train | Valid | Train report | Eval report |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fold in folds:
        spec = fold_specs[fold]
        row = by_fold.get(fold, {})
        train_link = rel_md_link(report_md.parent, row.get("train_report"))
        eval_link = rel_md_link(report_md.parent, row.get("eval_report"))
        train_cell = f"[report.md]({train_link})" if train_link else "—"
        eval_cell = f"[report.md]({eval_link})" if eval_link else "—"
        if row.get("eval_report"):
            idx = rel_md_link(report_md.parent, Path(row["eval_report"]).parent / "index.html")
            if idx:
                eval_cell += f" · [index]({idx})"
        lines.append(
            f"| {fold} | {', '.join(spec['train'])} | {', '.join(spec['valid'])} "
            f"| {train_cell} | {eval_cell} |"
        )

    lines.extend(
        [
            "",
            "## Comparison (eval, after lateral clean)",
            "",
            "| Fold | Valid | Train HR@1 | Eval HR@1 | HR@5 | MAE | RMSE | P90 | MBE | Replaced |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for fold in folds:
        spec = fold_specs[fold]
        row = by_fold.get(fold, {})
        metrics = dict(row.get("eval_metrics") or {})
        lateral = dict(metrics.get("lateral_clean") or {})
        replaced = lateral.get("n_replaced")
        lines.append(
            "| {fold} | {valid} | {thr} | {hr1} | {hr5} | {mae} | {rmse} | {p90} | {mbe} | {rep} |".format(
                fold=fold,
                valid=", ".join(spec["valid"]),
                thr=_fmt(row.get("train_hitrate1")),
                hr1=_fmt(metrics.get("HitRate1px")),
                hr5=_fmt(metrics.get("HitRate5px")),
                mae=_fmt(metrics.get("MeanAbsoluteError")),
                rmse=_fmt(metrics.get("RootMeanSquaredError")),
                p90=_fmt(metrics.get("P90AbsoluteError")),
                mbe=_fmt(metrics.get("MeanBiasError")),
                rep=_fmt(replaced),
            )
        )

    for fold in folds:
        row = by_fold.get(fold, {})
        spec = fold_specs[fold]
        lines.extend(["", f"## Fold {fold}", ""])
        if row.get("error"):
            lines.append(f"**Error:** {row['error']}")
            lines.append("")
        lines.append(f"- Train sites: {', '.join(spec['train'])}")
        lines.append(f"- Valid site: {', '.join(spec['valid'])}")
        if row.get("experiment_dir"):
            lines.append(f"- Experiment dir: `{row['experiment_dir']}`")
        if row.get("ckpt"):
            lines.append(f"- Best checkpoint: `{row['ckpt']}`")
        train_link = rel_md_link(report_md.parent, row.get("train_report"))
        eval_link = rel_md_link(report_md.parent, row.get("eval_report"))
        if train_link:
            lines.append(f"- [Training report]({train_link})")
        if eval_link:
            eval_dir = Path(row["eval_report"]).parent
            extras = [f"[eval report]({eval_link})"]
            for label, name in (
                ("index", "index.html"),
                ("worst gathers", "worst.html"),
                ("typical gathers", "typical.html"),
                ("stats", "stats.html"),
            ):
                href = rel_md_link(report_md.parent, eval_dir / name)
                if href:
                    extras.append(f"[{label}]({href})")
            lines.append("- " + " · ".join(extras))
        metrics = dict(row.get("eval_metrics") or {})
        if metrics:
            lines.extend(["", "| Metric | Value |", "| --- | --- |"])
            for key in _HEADLINE:
                if key in metrics:
                    lines.append(f"| {key} | {_fmt(metrics[key])} |")

    lines.extend(["", f"<!-- stamp {stamp} -->", ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return report_md


def _parse_marked_path(line: str, prefix: str) -> Path | None:
    text = line.strip()
    if not text.startswith(prefix):
        return None
    raw = text[len(prefix) :].strip()
    if not raw:
        return None
    return Path(raw)


def _run_logged(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    dry_run: bool,
) -> tuple[int, Path | None, Path | None]:
    """Run *cmd*, tee output, return ``(rc, experiment_dir, report_md)``."""
    experiment_dir: Path | None = None
    report_md: Path | None = None
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return 0, None, None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            found = _parse_marked_path(line, "Experiment dir:")
            if found is not None:
                experiment_dir = found
            found = _parse_marked_path(line, "Report:")
            if found is not None:
                report_md = found
        rc = proc.wait()
    return rc, experiment_dir, report_md


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _train_hitrate(experiment_dir: Path | None) -> float | None:
    if experiment_dir is None:
        return None
    manifest = Path(experiment_dir) / "best_checkpoint.json"
    if not manifest.is_file():
        return None
    try:
        score = _load_json(manifest).get("score")
        return float(score) if score is not None else None
    except (TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


def _eval_metrics(eval_report: Path | None) -> dict[str, Any]:
    if eval_report is None:
        return {}
    metrics_path = Path(eval_report).parent / "metrics.json"
    if not metrics_path.is_file():
        return {}
    try:
        data = _load_json(metrics_path)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and lateral-clean-eval folds A–D, then write a linked summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--folds",
        default="A-D",
        help="Fold letters: A-D, A,C, or A B D (leave-one-site-out only).",
    )
    p.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    p.add_argument("--backend", choices=("hdf5", "npz"), default="hdf5")
    p.add_argument("--picker", choices=(PICKER_FBPUNET, PICKER_BEFORE_AFTER), default=PICKER_BEFORE_AFTER)
    p.add_argument("--geonorm", default="D", help="GeoNorm ablation passed to train (A/B/C/D).")
    p.add_argument("--config", type=Path, default=None, help="Training recipe YAML.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--devices", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Root for train/eval/summary reports (default: <repo>/report).",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; evaluate existing fold checkpoints and write a new summary.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to look for train_foldX_* dirs in --eval-only (default: <repo>/output).",
    )
    p.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="Prior sweep.json whose experiment_dir/ckpt paths are reused in --eval-only.",
    )
    p.add_argument(
        "--ckpt-dir",
        action="append",
        default=[],
        metavar="FOLD=DIR",
        help="Fold experiment directory for --eval-only (repeatable). A bare path is ok if --folds has one letter.",
    )
    p.add_argument(
        "--before-after-decoder",
        choices=BEFORE_AFTER_DECODERS,
        default=None,
        help=(
            "Eval-only override of how before/after logits become picks "
            "(legacy|change_point). Default: checkpoint, or legacy if unset."
        ),
    )
    p.add_argument("--lateral-clean", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lateral-window", type=int, default=DEFAULT_LATERAL_WINDOW)
    p.add_argument("--lateral-max-dev", type=float, default=DEFAULT_LATERAL_MAX_DEV)
    p.add_argument("--lateral-max-flag-frac", type=float, default=DEFAULT_LATERAL_MAX_FLAG_FRAC)
    p.add_argument("--lateral-min-anchors", type=int, default=DEFAULT_LATERAL_MIN_ANCHORS)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "train_passthrough",
        nargs=argparse.REMAINDER,
        help="After --, extra arguments forwarded only to fbp_train.py.",
    )
    args = p.parse_args(argv)
    args.folds = parse_folds(args.folds)
    extra = list(args.train_passthrough or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    args.train_passthrough = extra
    if extra and args.eval_only:
        p.error("--eval-only does not take extra fbp_train.py arguments after --")
    if args.before_after_decoder and args.picker != PICKER_BEFORE_AFTER:
        p.error("--before-after-decoder requires --picker before_after")
    try:
        args.ckpt_dirs = parse_ckpt_dirs(args.ckpt_dir, args.folds)
    except ValueError as exc:
        p.error(str(exc))
    args.sweep_data = None
    if args.sweep is not None:
        sweep_path = Path(args.sweep).expanduser().resolve()
        if not sweep_path.is_file():
            p.error(f"--sweep is not a file: {sweep_path}")
        payload = _load_json(sweep_path)
        args.sweep_data = payload if isinstance(payload, dict) else {}
    return args


def _train_cmd(args: argparse.Namespace, fold: str, report_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train" / "fbp_train.py"),
        "--fold",
        fold,
        "--picker",
        args.picker,
        "--geonorm",
        str(args.geonorm),
        "--backend",
        args.backend,
        "--data-dir",
        str(args.data_dir),
        "--report-dir",
        str(report_root),
    ]
    if args.config:
        cmd.extend(["--config", str(args.config)])
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.patience is not None:
        cmd.extend(["--patience", str(args.patience)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.devices is not None:
        cmd.extend(["--devices", str(args.devices)])
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    cmd.extend(args.train_passthrough)
    return cmd


def _eval_cmd(
    args: argparse.Namespace,
    fold: str,
    ckpt: Path,
    report_root: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "train" / "fbp_eval.py"),
        "--picker",
        args.picker,
        "--ckpt",
        str(ckpt),
        "--fold",
        fold,
        "--backend",
        args.backend,
        "--data-dir",
        str(args.data_dir),
        "--report-dir",
        str(report_root),
    ]
    if args.num_workers is not None:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.before_after_decoder:
        cmd.extend(["--before-after-decoder", args.before_after_decoder])
    if args.lateral_clean:
        cmd.append("--lateral-clean")
        cmd.extend(["--lateral-window", str(args.lateral_window)])
        cmd.extend(["--lateral-max-dev", str(args.lateral_max_dev)])
        cmd.extend(["--lateral-max-flag-frac", str(args.lateral_max_flag_frac)])
        cmd.extend(["--lateral-min-anchors", str(args.lateral_min_anchors)])
    return cmd


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_root = (args.report_root or (REPO_ROOT / "report")).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = "".join(args.folds) if args.folds != list(DEFAULT_FOLDS) else "AD"
    if args.eval_only:
        label = f"{label}_eval"
    summary_dir = report_root / f"folds_{label}_{stamp}"
    summary_md = summary_dir / "report.md"
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, Any]] = []
    output_root = (args.output_root or (REPO_ROOT / "output")).resolve()
    meta: dict[str, Any] = {
        "label": "–".join(args.folds) if len(args.folds) > 1 else args.folds[0],
        "status": "running",
        "started": started,
        "finished": None,
        "stamp": stamp,
        "picker": args.picker,
        "geonorm": args.geonorm,
        "backend": args.backend,
        "lateral_clean": bool(args.lateral_clean),
        "data_dir": str(Path(args.data_dir).resolve()),
        "eval_only": bool(args.eval_only),
        "before_after_decoder": args.before_after_decoder,
    }

    def flush(status: str, finished: str | None = None) -> None:
        meta["status"] = status
        meta["finished"] = finished
        write_folds_summary(
            summary_md,
            folds=args.folds,
            fold_specs=FOLDS_AD,
            rows=rows,
            meta=meta,
        )
        (summary_dir / "sweep.json").write_text(
            json.dumps({"meta": meta, "folds": rows}, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    summary_dir.mkdir(parents=True, exist_ok=True)
    flush("running")
    print(f"Summary: {summary_md}", flush=True)

    failed = False
    for fold in args.folds:
        spec = FOLDS_AD[fold]
        row: dict[str, Any] = {
            "fold": fold,
            "train_sites": list(spec["train"]),
            "valid_sites": list(spec["valid"]),
            "error": None,
            "experiment_dir": None,
            "train_report": None,
            "eval_report": None,
            "ckpt": None,
            "train_hitrate1": None,
            "eval_metrics": {},
        }
        rows.append(row)
        print(f"\n======== Fold {fold}  train {spec['train']}  valid {spec['valid']} ========", flush=True)
        if args.eval_only:
            try:
                exp_dir = find_fold_experiment(
                    fold,
                    output_root=output_root,
                    explicit=args.ckpt_dirs,
                    sweep=args.sweep_data,
                )
            except FileNotFoundError as exc:
                row["error"] = str(exc)
                failed = True
                flush("running")
                if not args.continue_on_error:
                    break
                continue
            row["experiment_dir"] = str(exp_dir)
            train_md = find_train_report(exp_dir, report_root)
            row["train_report"] = str(train_md) if train_md else None
            print(f"Reusing experiment dir: {exp_dir}", flush=True)
        else:
            train_log = summary_dir / f"fold{fold}_train.log"
            rc, exp_dir, train_report = _run_logged(
                _train_cmd(args, fold, report_root),
                cwd=REPO_ROOT,
                log_path=train_log,
                dry_run=args.dry_run,
            )
            row["experiment_dir"] = str(exp_dir) if exp_dir else None
            row["train_report"] = str(train_report) if train_report else None
            if rc != 0:
                row["error"] = f"train exited {rc} (log: {train_log})"
                failed = True
                flush("running")
                if not args.continue_on_error:
                    break
                continue
        if args.dry_run:
            if args.eval_only and exp_dir is not None:
                try:
                    from seismic_utils.predict import resolve_checkpoint

                    preview = resolve_checkpoint(ckpt_dir=exp_dir)
                except FileNotFoundError:
                    preview = exp_dir / "best*.ckpt"
                print("+", " ".join(_eval_cmd(args, fold, Path(preview), report_root)), flush=True)
            flush("running")
            continue
        try:
            from seismic_utils.predict import resolve_checkpoint

            ckpt = resolve_checkpoint(ckpt_dir=exp_dir)
        except FileNotFoundError as exc:
            row["error"] = str(exc)
            failed = True
            flush("running")
            if not args.continue_on_error:
                break
            continue
        row["ckpt"] = str(ckpt)
        row["train_hitrate1"] = _train_hitrate(exp_dir)
        eval_log = summary_dir / f"fold{fold}_eval.log"
        rc, _, eval_report = _run_logged(
            _eval_cmd(args, fold, ckpt, report_root),
            cwd=REPO_ROOT,
            log_path=eval_log,
            dry_run=False,
        )
        row["eval_report"] = str(eval_report) if eval_report else None
        row["eval_metrics"] = _eval_metrics(eval_report)
        if rc != 0:
            row["error"] = f"eval exited {rc} (log: {eval_log})"
            failed = True
            flush("running")
            if not args.continue_on_error:
                break
            continue
        flush("running")

    finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.dry_run:
        flush("dry-run", finished)
    elif failed:
        flush("failed", finished)
    else:
        flush("finished", finished)
    print(f"\nSummary report: {summary_md}", flush=True)
    return 1 if failed and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
