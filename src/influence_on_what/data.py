"""Load training data for influence function experiments.

Supports the following folder structure:

    data_training/
        base/*.txt          → D_base  (category="base")
        X/*.txt     → D_X     (category="X")
    data_test/
        base/*.txt          → held-out base sentences (PPL sanity check)
        X/good/*.txt
        X/bad/*.txt → minimal pairs (loaded by queries.py)

Training datasets are condition-specific:
  full       → D_base ∪ D_X
  base_only  → D_base only

By default, rows are shuffled within each category block before those blocks are
placed according to the configured mixing strategy.

No internal train/test split is performed — the test set lives entirely in
data_test/ and is loaded separately by queries.py.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Generator

import datasets

from .config import ExperimentCfg

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────


def _iter_files(paths: list[Path], category: str) -> Generator[dict, None, None]:
    """Stream non-empty lines from multiple files as dataset rows. O(1) memory."""
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    yield {"text": line, "category": category}


def _reservoir_sample(
    paths: list[Path], category: str, max_samples: int, seed: int
) -> list[dict]:
    """Reservoir-sample up to max_samples rows across files. O(max_samples) peak memory."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    count = 0
    for row in _iter_files(paths, category):
        if len(reservoir) < max_samples:
            reservoir.append(row)
        else:
            j = rng.randint(0, count)
            if j < max_samples:
                reservoir[j] = row
        count += 1
    return reservoir


def _repeat_to_fill(rows: list[dict], target: int) -> list[dict]:
    """Repeat rows in full rounds until target is reached, then truncate."""
    result: list[dict] = []
    while len(result) < target:
        result.extend(rows)
    return result[:target]


def _build_category_dataset(
    paths: list[Path], category: str, max_samples: int, seed: int, oversample: bool = False
) -> datasets.Dataset:
    """Build one category dataset, optionally with deterministic subsampling."""

    def gen() -> Generator[dict, None, None]:
        if max_samples > 0:
            rows = _reservoir_sample(paths, category, max_samples, seed)
            if oversample and len(rows) < max_samples:
                rows = _repeat_to_fill(rows, max_samples)
            yield from rows
        else:
            yield from _iter_files(paths, category)

    return datasets.Dataset.from_generator(gen)


def _interleave_shuffled_X_into_base(
    base_ds: datasets.Dataset, X_ds: datasets.Dataset, seed: int
) -> datasets.Dataset:
    """Randomly interleave X into base while preserving base order."""
    rng = random.Random(seed)
    shuffled_x = list(X_ds)
    rng.shuffle(shuffled_x)

    x_by_gap: list[list[dict]] = [[] for _ in range(len(base_ds) + 1)]
    for row in shuffled_x:
        gap_idx = rng.randint(0, len(base_ds))
        x_by_gap[gap_idx].append(row)

    def gen() -> Generator[dict, None, None]:
        for row in x_by_gap[0]:
            yield row
        for idx, base_row in enumerate(base_ds):
            yield base_row
            for row in x_by_gap[idx + 1]:
                yield row

    return datasets.Dataset.from_generator(gen)


# ── training dataset assembly ────────────────────────────────────────────────


def _build_training_dataset(
    data_dir: Path,
    condition: str,
    max_samples_base: int,
    max_samples_X: int,
    oversample_X: bool,
    shuffle_within_blocks: bool,
    mixing_strategy: str,
    seed: int,
) -> datasets.Dataset:
    """Build the training dataset for the given condition.

    Returns a Dataset with columns ``text`` and ``category``.
    """
    base_dir = data_dir / "data_training" / "base"
    if not base_dir.exists():
        raise FileNotFoundError(f"Base training directory not found: {base_dir}")
    base_files = sorted(base_dir.glob("*.txt"))
    if not base_files:
        raise FileNotFoundError(f"No .txt files found in: {base_dir}")

    X_files: list[Path] = []
    if condition == "full":
        X_dir = data_dir / "data_training" / "X"
        if not X_dir.exists():
            raise FileNotFoundError(f"D_X directory not found: {X_dir}")
        X_files = sorted(X_dir.glob("*.txt"))

    base_ds = _build_category_dataset(base_files, "base", max_samples_base, seed)
    if shuffle_within_blocks:
        base_ds = base_ds.shuffle(seed=seed)

    if condition != "full":
        return base_ds

    X_ds = _build_category_dataset(X_files, "X", max_samples_X, seed, oversample=oversample_X)
    if shuffle_within_blocks:
        X_ds = X_ds.shuffle(seed=seed + 1)

    if mixing_strategy == "start":
        return datasets.concatenate_datasets([X_ds, base_ds])

    if mixing_strategy == "end":
        return datasets.concatenate_datasets([base_ds, X_ds])

    if mixing_strategy == "middle":
        midpoint = len(base_ds) // 2
        base_head = base_ds.select(range(midpoint))
        base_tail = base_ds.select(range(midpoint, len(base_ds)))
        return datasets.concatenate_datasets([base_head, X_ds, base_tail])

    if mixing_strategy == "full_shuffle":
        return _interleave_shuffled_X_into_base(base_ds, X_ds, seed)

    raise ValueError(
        "Unknown data.mixing_strategy: "
        f"{mixing_strategy!r}. Choose from start, end, middle, full_shuffle."
    )


def _save_base_test(data_dir: Path, out_path: Path) -> None:
    """Save the held-out base corpus sentences for PPL sanity checks."""
    test_dir = data_dir / "data_test" / "base"
    test_files = sorted(test_dir.glob("*.txt")) if test_dir.exists() else []
    if not test_files:
        logger.warning(f"No .txt files found in {test_dir}, skipping.")
        return

    def gen():
        yield from _iter_files(test_files, "base")

    ds = datasets.Dataset.from_generator(gen)
    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))
    logger.info(f"  Base test: {len(ds):,} samples → {out_path}")


# ── public API ───────────────────────────────────────────────────────────────


def prepare_and_save(exp: ExperimentCfg) -> None:
    """Load and save training data for the current condition.

    Saves:
      artifacts/<name>/shared/train_<condition>/   — training dataset
      artifacts/<name>/shared/base_test/           — base test set (once)
    """
    shared = exp.shared_dir()
    shared.mkdir(parents=True, exist_ok=True)

    seed = exp.seeds[0]
    data_dir = Path(exp.data.data_dir)

    logger.info(
        "Preparing training data: condition=%s, mixing_strategy=%s, "
        "shuffle_within_blocks=%s, data_dir=%s",
        exp.data.condition,
        exp.data.mixing_strategy,
        exp.data.shuffle_within_blocks,
        data_dir,
    )

    train_ds = _build_training_dataset(
        data_dir=data_dir,
        condition=exp.data.condition,
        max_samples_base=exp.data.max_samples_base,
        max_samples_X=exp.data.max_samples_X,
        oversample_X=exp.data.oversample_X,
        shuffle_within_blocks=exp.data.shuffle_within_blocks,
        mixing_strategy=exp.data.mixing_strategy,
        seed=seed,
    )

    train_path = exp.train_data_path()
    train_path.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(str(train_path))
    n_base = sum(c == "base" for c in train_ds["category"])
    n_X = len(train_ds) - n_base
    logger.info(
        f"Saved {len(train_ds):,} training samples → {train_path}  "
        f"(base={n_base:,}, X={n_X:,})"
    )

    # Save base test set once (shared across conditions).
    base_test_path = exp.base_test_path()
    if not base_test_path.exists():
        _save_base_test(data_dir, base_test_path)
