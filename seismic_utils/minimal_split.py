"""Site-specific 1% labelled splits for the Meneses minimal-annotation recipe.

Does not use hardpicks ``eval_ratio`` (shot-and-line holdout). Keys are
``(origin, gather_id, shot_id, rec_line_id)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PAPER_LABELED_COUNTS: dict[str, int] = {
    "Brunswick": 148,
    "Halfmile": 54,
    "Lalor": 120,
    "Sudbury": 43,
}

MIN_ANNOTATION_FRAC = 0.01
LABELED_TRAIN_FRAC = 0.75
BAD_PICK_INDEX = 0

GatherKey = tuple[str, int, int, int]


def gather_key(meta: Mapping[str, Any]) -> GatherKey:
    origin = str(meta.get("origin") or meta.get("site_name") or "")
    return (
        origin,
        int(meta["gather_id"]),
        int(meta["shot_id"]),
        int(meta.get("rec_line_id", -1)),
    )


def key_to_dict(key: GatherKey) -> dict[str, Any]:
    origin, gather_id, shot_id, rec_line_id = key
    return {
        "origin": origin,
        "gather_id": int(gather_id),
        "shot_id": int(shot_id),
        "rec_line_id": int(rec_line_id),
    }


def dict_to_key(row: Mapping[str, Any]) -> GatherKey:
    return (
        str(row["origin"]),
        int(row["gather_id"]),
        int(row["shot_id"]),
        int(row.get("rec_line_id", -1)),
    )


def labeled_fraction(meta: Mapping[str, Any]) -> float:
    mask = meta.get("bad_first_breaks_mask")
    labels = meta.get("first_break_labels")
    if mask is not None:
        bad = np.asarray(mask, dtype=bool).reshape(-1)
        n = int(bad.size)
        if n == 0:
            return 0.0
        return float((~bad).sum()) / float(n)
    if labels is None:
        return 0.0
    labs = np.asarray(labels).reshape(-1)
    n = int(labs.size)
    if n == 0:
        return 0.0
    return float((labs > BAD_PICK_INDEX).sum()) / float(n)


def is_valid_gather(meta: Mapping[str, Any], min_frac: float = MIN_ANNOTATION_FRAC) -> bool:
    """Keep gathers with at least ``min_frac`` annotated traces."""
    return labeled_fraction(meta) >= float(min_frac)


def collect_valid_keys(
    parser,
    *,
    min_frac: float = MIN_ANNOTATION_FRAC,
) -> list[GatherKey]:
    keys: list[GatherKey] = []
    for i in range(len(parser)):
        meta = parser.get_meta_gather(i)
        if is_valid_gather(meta, min_frac=min_frac):
            keys.append(gather_key(meta))
    return keys


def index_map(parser) -> dict[GatherKey, int]:
    out: dict[GatherKey, int] = {}
    for i in range(len(parser)):
        out[gather_key(parser.get_meta_gather(i))] = i
    return out


def indices_for_keys(parser, keys: Sequence[GatherKey]) -> list[int]:
    lookup = index_map(parser)
    missing = [k for k in keys if k not in lookup]
    if missing:
        raise KeyError(f"{len(missing)} split key(s) not in parser, e.g. {missing[0]}")
    return [lookup[k] for k in keys]


def paper_labeled_count(site: str) -> int:
    key = site.strip()
    if key not in PAPER_LABELED_COUNTS:
        known = ", ".join(PAPER_LABELED_COUNTS)
        raise KeyError(f"Unknown site {site!r}; expected one of: {known}")
    return int(PAPER_LABELED_COUNTS[key])


def _split_labeled(keys: Sequence[GatherKey], rng: np.random.Generator) -> tuple[list[GatherKey], list[GatherKey]]:
    keys = list(keys)
    n = len(keys)
    if n < 2:
        raise ValueError("need at least 2 labelled gathers for a 75/25 split")
    n_train = int(round(n * LABELED_TRAIN_FRAC))
    n_train = min(max(n_train, 1), n - 1)
    order = rng.permutation(n)
    train = [keys[i] for i in order[:n_train]]
    valid = [keys[i] for i in order[n_train:]]
    return train, valid


def make_minimal_split(
    valid_keys: Sequence[GatherKey],
    *,
    site: str,
    seed: int,
    n_labeled: int | None = None,
) -> dict[str, Any]:
    """Draw the paper-sized labelled pool (or ``n_labeled``) and a 75/25 split."""
    keys = list(valid_keys)
    n_want = int(n_labeled) if n_labeled is not None else paper_labeled_count(site)
    if n_want < 2:
        raise ValueError("n_labeled must be >= 2")
    if len(keys) < n_want:
        raise ValueError(
            f"{site}: {len(keys)} valid gathers < requested labelled count {n_want}"
        )
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(keys))
    labeled = [keys[i] for i in order[:n_want]]
    unlabeled = [keys[i] for i in order[n_want:]]
    labeled_train, labeled_val = _split_labeled(labeled, rng)
    return {
        "site": site,
        "seed": int(seed),
        "n_valid": len(keys),
        "n_labeled": n_want,
        "labeled_train": [key_to_dict(k) for k in labeled_train],
        "labeled_val": [key_to_dict(k) for k in labeled_val],
        "unlabeled_pool": [key_to_dict(k) for k in unlabeled],
    }


def split_keys(split: Mapping[str, Any], field: str) -> list[GatherKey]:
    return [dict_to_key(row) for row in split.get(field) or []]


def write_split(split: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=2) + "\n")
    return path


def load_split(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"split file must be a mapping: {path}")
    return data


def assert_unlabeled_item_has_no_gt(item: Mapping[str, Any]) -> None:
    """CI guard: train unlabeled / stripped views must not expose finite GT picks."""
    labels = np.asarray(item.get("first_break_labels", []), dtype=np.float64).reshape(-1)
    if labels.size and np.any(labels > BAD_PICK_INDEX):
        raise AssertionError("unlabeled item still contains ground-truth first-break labels")
    mask = item.get("segmentation_mask")
    if mask is None:
        return
    arr = np.asarray(mask)
    allowed = {0, -1}
    uniq = set(np.unique(arr).tolist())
    if uniq - allowed:
        raise AssertionError(f"unlabeled mask has unexpected labels {sorted(uniq - allowed)}")
    if 1 in uniq:
        raise AssertionError("unlabeled mask still contains first-break class pixels")
