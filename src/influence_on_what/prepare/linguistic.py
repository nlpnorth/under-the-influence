#!/usr/bin/env python3
"""Prepare query and training datasets for goldfish BLiMP attribution.

Creates HF datasets at the paths expected by `python -m influence_on_what attribute`:
  shared/queries/   — BLiMP minimal pairs (feature-specific); 'npi_licensing' is
                      served from the manual NPI TSV instead, see --manual-npi-tsv
  shared/train/     — D_base (clean) + D_X (matched) with category labels
  <tag>/seed_<seed>/model — symlink to pretrained goldfish model

Sampling: D_base is ratio-proportionally sampled from clean_file across chunks
(proportional to each chunk's line count).  D_X is loaded in full from
matched_file (typically small: a few thousand sentences per feature).

Run once per (model, feature) combination before `python -m influence_on_what attribute`.

Examples
--------
# Full model — binding feature as D_X
python prepare/linguistic.py \\
    --config configs/goldfish_100m_attribution.yaml \\
    --name goldfish_100m_full_binding_blimp \\
    --model-path /path/to/models/if_goldfish_100m_full \\
    --data-root /home/argy/english_data/filter_output \\
    --chunks 00 01 \\
    --clean-file "Binding-reflexive/train_clean" \\
    --matched-file "Binding-reflexive/train_matched" \\
    --phenomena '{"binding": ["principle_A_c_command"]}' \\
    --max-base 5000000

# Ablated model — binding feature removed; matched sentences still scored (as absent D_X)
python prepare/linguistic.py \\
    --name goldfish_100m_no_binding_reflexive_blimp \\
    --clean-file "Binding-reflexive/train_clean" \\
    --matched-file "Binding-reflexive/train_matched" \\
    ...
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import urllib.request
from pathlib import Path

import datasets
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[prep %(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BLIMP_BASE = "https://raw.githubusercontent.com/alexwarstadt/blimp/master/data"


# ── BLiMP helpers ─────────────────────────────────────────────────────────────


def ensure_blimp_files(blimp_dir: Path, phenomena: dict[str, list[str]]) -> None:
    """Download missing BLiMP .jsonl files."""
    blimp_dir.mkdir(parents=True, exist_ok=True)
    for phenomenon, subitems in phenomena.items():
        if phenomenon == "npi_licensing":
            continue  # served from the manual NPI TSV, not BLiMP
        for sub in subitems:
            dest = blimp_dir / f"{sub}.jsonl"
            if not dest.exists():
                url = f"{BLIMP_BASE}/{sub}.jsonl"
                logger.info("Downloading %s → %s", url, dest)
                urllib.request.urlretrieve(url, dest)


def load_manual_npi_queries(tsv_path: Path) -> tuple[list[str], list[str], list[str]]:
    """Load NPI minimal pairs from the manual TSV (see probe_models.py's
    load_manual_npi_pairs). Uses the 'env' column as the feature label."""
    if not tsv_path.exists():
        raise FileNotFoundError(f"Manual NPI TSV not found: {tsv_path}")
    good: list[str] = []
    bad: list[str] = []
    features: list[str] = []
    with open(tsv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            good.append(row["sen"])
            bad.append(row["wrong_sen"])
            features.append(row.get("env", "manual_npi"))
    return good, bad, features


def build_queries(
    blimp_dir: Path,
    phenomena: dict[str, list[str]],
    manual_npi_tsv: Path | None = None,
) -> datasets.Dataset:
    """Load minimal pairs for the specified phenomena/subitems into a queries dataset.

    The 'npi_licensing' phenomenon is served from the manual NPI TSV instead of
    BLiMP; all other phenomena are loaded from BLiMP .jsonl files as before.
    """
    all_good: list[str] = []
    all_bad: list[str] = []
    all_features: list[str] = []

    for phenomenon, subitems in sorted(phenomena.items()):
        if phenomenon == "npi_licensing":
            if manual_npi_tsv is None:
                raise ValueError(
                    "phenomena includes 'npi_licensing' but --manual-npi-tsv was not provided"
                )
            good, bad, features = load_manual_npi_queries(manual_npi_tsv)
            all_good.extend(good)
            all_bad.extend(bad)
            all_features.extend(features)
            logger.info("  [%s] manual NPI TSV: %d pairs", phenomenon, len(good))
            continue

        for sub in subitems:
            path = blimp_dir / f"{sub}.jsonl"
            pairs: list[tuple[str, str]] = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    pairs.append((entry["sentence_good"], entry["sentence_bad"]))
            all_good.extend(g for g, _ in pairs)
            all_bad.extend(b for _, b in pairs)
            all_features.extend([sub] * len(pairs))
            logger.info("  [%s] %s: %d pairs", phenomenon, sub, len(pairs))

    logger.info("Total: %d query pairs across %d phenomena", len(all_good), len(phenomena))
    return datasets.Dataset.from_dict(
        {"text_good": all_good, "text_bad": all_bad, "feature": all_features}
    )


def stratified_sample_queries(
    queries: datasets.Dataset, n: int, labels_parquet: str, seed: int
) -> datasets.Dataset:
    """Subsample *queries* to at most *n* rows, split as evenly as possible
    between pairs probe_models.py's full model already learned
    (learned_x=True, i.e. softmax_prob_good > 0.5) and pairs it hasn't — so a
    small sample isn't skewed entirely to one side. Matches queries to labels
    by exact (text_good, text_bad) content (see attribute.py's
    _align_pairs_parquet, which does the same join downstream). Queries with
    no matching label row are sampled uniformly as a last-resort filler.
    """
    labels_df = pd.read_parquet(
        labels_parquet, columns=["text_good", "text_bad", "learned_x"]
    )
    learned_by_pair = dict(
        zip(zip(labels_df["text_good"], labels_df["text_bad"]), labels_df["learned_x"])
    )

    learned_idx: list[int] = []
    not_learned_idx: list[int] = []
    unknown_idx: list[int] = []
    for i, (g, b) in enumerate(zip(queries["text_good"], queries["text_bad"])):
        label = learned_by_pair.get((g, b))
        if label is None:
            unknown_idx.append(i)
        elif label:
            learned_idx.append(i)
        else:
            not_learned_idx.append(i)

    if unknown_idx:
        logger.warning(
            "%d/%d queries have no matching row in %s — excluded from the "
            "learned/not-learned split, sampled uniformly as filler only",
            len(unknown_idx), len(queries), labels_parquet,
        )
    if not learned_idx or not not_learned_idx:
        logger.warning(
            "Only one learned-status group is non-empty (%d learned, %d "
            "not-learned) — cannot guarantee both appear in the sample.",
            len(learned_idx), len(not_learned_idx),
        )

    rng = random.Random(seed)
    rng.shuffle(learned_idx)
    rng.shuffle(not_learned_idx)
    rng.shuffle(unknown_idx)

    target_learned = n // 2
    target_not_learned = n - target_learned
    take_learned = min(target_learned, len(learned_idx))
    take_not_learned = min(target_not_learned, len(not_learned_idx))

    # Backfill any deficit (one group too small) from the other group's leftover.
    deficit = n - take_learned - take_not_learned
    if deficit > 0:
        extra = min(deficit, len(learned_idx) - take_learned)
        take_learned += extra
        deficit -= extra
    if deficit > 0:
        extra = min(deficit, len(not_learned_idx) - take_not_learned)
        take_not_learned += extra
        deficit -= extra

    selected = learned_idx[:take_learned] + not_learned_idx[:take_not_learned]
    if deficit > 0:
        selected += unknown_idx[:deficit]

    logger.info(
        "Stratified sample: %d learned + %d not-learned + %d unknown = %d "
        "(of %d/%d/%d available)",
        take_learned, take_not_learned, len(selected) - take_learned - take_not_learned,
        len(selected), len(learned_idx), len(not_learned_idx), len(unknown_idx),
    )
    return queries.select(sorted(selected))


def cascade_sample_queries(
    queries: datasets.Dataset,
    n: int,
    full_pairs_parquet: str,
    no_x_pairs_parquet: str,
    seed: int,
    retained_target: int = 30,
    not_learned_target: int = 30,
) -> datasets.Dataset:
    """Subsample *queries* to at most *n* rows via a 3-way cascade over two
    probe_models.py --pairs-output parquets: *full_pairs_parquet* (the true
    full model, trained on D_base ∪ D_X) and *no_x_pairs_parquet* (the ablated
    model under attribution, trained on D_base only). "Learned" means
    softmax_prob_good > 0.5 (p(correct) > p(incorrect)), matching
    LEARNED_THRESHOLD in attribute.py.

    Three mutually exclusive categories (naming mirrors analyze.py's bear-facts
    Retained/Forgotten/not_learned split):
      retained         — full model learned it AND the ablated model also did
                          (kept it despite the ablation — not X-specific)
      learned_specific  — full model learned it, ablated model did NOT
                          (the ablation actually removed this knowledge)
      not_learned       — full model never learned it (regardless of ablated model)

    Cascade: take up to *retained_target* from 'retained', up to
    *not_learned_target* from 'not_learned', then fill the rest of the *n*
    budget from 'learned_specific'. No backfill between categories beyond
    that — if a category runs short, the sample is simply smaller than *n*.

    Matched to queries by exact (text_good, text_bad) content (see
    attribute.py's _align_pairs_parquet, which does the same join
    downstream). Queries missing from either parquet are excluded entirely.
    """
    full_df = pd.read_parquet(
        full_pairs_parquet, columns=["text_good", "text_bad", "learned_x"]
    )
    no_x_df = pd.read_parquet(
        no_x_pairs_parquet, columns=["text_good", "text_bad", "learned_x"]
    )
    full_by_pair = dict(
        zip(zip(full_df["text_good"], full_df["text_bad"]), full_df["learned_x"])
    )
    no_x_by_pair = dict(
        zip(zip(no_x_df["text_good"], no_x_df["text_bad"]), no_x_df["learned_x"])
    )

    retained_idx: list[int] = []
    not_learned_idx: list[int] = []
    learned_specific_idx: list[int] = []
    unknown_idx: list[int] = []
    for i, (g, b) in enumerate(zip(queries["text_good"], queries["text_bad"])):
        learned_full = full_by_pair.get((g, b))
        learned_no_x = no_x_by_pair.get((g, b))
        if learned_full is None or learned_no_x is None:
            unknown_idx.append(i)
        elif not learned_full:
            not_learned_idx.append(i)
        elif learned_no_x:
            retained_idx.append(i)
        else:
            learned_specific_idx.append(i)

    if unknown_idx:
        logger.warning(
            "%d/%d queries missing from full/no-x pairs parquets — excluded "
            "from the cascade sample",
            len(unknown_idx), len(queries),
        )

    rng = random.Random(seed)
    rng.shuffle(retained_idx)
    rng.shuffle(not_learned_idx)
    rng.shuffle(learned_specific_idx)

    take_retained = min(retained_target, len(retained_idx))
    take_not_learned = min(not_learned_target, len(not_learned_idx))
    remaining = max(n - take_retained - take_not_learned, 0)
    take_specific = min(remaining, len(learned_specific_idx))

    selected = (
        retained_idx[:take_retained]
        + not_learned_idx[:take_not_learned]
        + learned_specific_idx[:take_specific]
    )
    logger.info(
        "Cascade sample: %d retained + %d not_learned + %d learned_specific "
        "= %d (of %d/%d/%d available; targets %d/%d/remainder)",
        take_retained, take_not_learned, take_specific, len(selected),
        len(retained_idx), len(not_learned_idx), len(learned_specific_idx),
        retained_target, not_learned_target,
    )
    if len(selected) < n:
        logger.warning(
            "Only %d/%d requested queries available across the three cascade "
            "categories.",
            len(selected), n,
        )
    return queries.select(sorted(selected))


# ── Training corpus helpers ────────────────────────────────────────────────────


def count_lines(path: Path) -> int:
    """Count non-empty lines in a text file."""
    n = 0
    with open(path, "rb") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def reservoir_sample(path: Path, n: int, seed: int) -> list[str]:
    """Reservoir-sample n non-empty lines from path. O(n) peak memory."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, seen)
                if j < n:
                    reservoir[j] = line
            seen += 1
    return reservoir


def build_training_data(
    data_root: Path,
    chunks: list[str],
    clean_file: str,
    matched_file: str | None,
    max_base: int,
    seed: int = 0,
) -> datasets.Dataset:
    """Build training dataset with D_base (category='base') and D_X (category='X').

    Both components are sampled within a total budget of max_base, preserving the
    natural D_X : D_base ratio from the full corpus:
      n_dx   = round(total_dx / (total_base + total_dx) * max_base), capped at total_dx
      n_base = min(max_base - n_dx, total_base)
    Each component is then distributed across chunks proportionally.
    """
    # ── D_base: count lines per chunk ─────────────────────────────────────────
    counts_base: dict[str, int] = {}
    for chunk in chunks:
        src = data_root / f"chunk_{chunk}" / f"{clean_file}.txt"
        if not src.exists():
            raise FileNotFoundError(f"Clean corpus not found: {src}")
        counts_base[chunk] = count_lines(src)
        logger.info("  chunk_%s: %d lines (D_base)", chunk, counts_base[chunk])
    total_base = sum(counts_base.values())

    # ── D_X: count lines per chunk ────────────────────────────────────────────
    counts_dx: dict[str, int] = {}
    total_dx = 0
    if matched_file is not None:
        for chunk in chunks:
            src = data_root / f"chunk_{chunk}" / f"{matched_file}.txt"
            if src.exists():
                counts_dx[chunk] = count_lines(src)
            else:
                logger.warning("Matched file not found: %s — skipping chunk_%s", src, chunk)
                counts_dx[chunk] = 0
        total_dx = sum(counts_dx.values())
        logger.info("D_X total: %d lines", total_dx)

    # ── Budget split preserving corpus ratio ──────────────────────────────────
    full_corpus_size = total_base + total_dx
    if max_base == -1:
        n_dx = total_dx
        n_base = total_base
    elif total_dx > 0:
        n_dx = min(round(total_dx / full_corpus_size * max_base), total_dx)
        n_base = min(max_base - n_dx, total_base)
    else:
        n_dx = 0
        n_base = min(max_base, total_base)
    logger.info(
        "Budget: %d total (D_base=%d, D_X=%d, D_X ratio=%.4f)",
        n_base + n_dx, n_base, n_dx,
        total_dx / full_corpus_size if full_corpus_size > 0 else 0.0,
    )

    # Stream rows through a generator instead of materializing a list of dicts —
    # at max_base=-1 (full 10b corpus, ~146M rows) the list-of-dicts approach
    # OOMs well before save_to_disk; from_generator writes to the Arrow cache
    # incrementally, keeping at most one chunk's reservoir sample in memory.
    actual_counts = {"base": 0, "X": 0}

    def _generate_rows():
        # ── D_base: reservoir-sample proportionally across chunks ─────────────
        for chunk in chunks:
            ratio = counts_base[chunk] / total_base
            n = min(round(ratio * n_base), counts_base[chunk])
            src = data_root / f"chunk_{chunk}" / f"{clean_file}.txt"
            lines = reservoir_sample(src, n, seed=seed + abs(hash(chunk)) % 100_000)
            logger.info(
                "  chunk_%s: sampled %d / %d D_base (ratio=%.3f)",
                chunk, len(lines), counts_base[chunk], ratio,
            )
            for line in lines:
                actual_counts["base"] += 1
                yield {"text": line, "category": "base"}

        # ── D_X: reservoir-sample proportionally across chunks ─────────────────
        if matched_file is not None and n_dx > 0:
            for chunk in chunks:
                if counts_dx.get(chunk, 0) == 0:
                    continue
                ratio = counts_dx[chunk] / total_dx
                n = min(round(ratio * n_dx), counts_dx[chunk])
                src = data_root / f"chunk_{chunk}" / f"{matched_file}.txt"
                lines = reservoir_sample(src, n, seed=seed + abs(hash("dx_" + chunk)) % 100_000)
                logger.info(
                    "  chunk_%s: sampled %d / %d D_X (ratio=%.3f)",
                    chunk, len(lines), counts_dx[chunk], ratio,
                )
                for line in lines:
                    actual_counts["X"] += 1
                    yield {"text": line, "category": "X"}

    train_ds = datasets.Dataset.from_generator(_generate_rows)
    logger.info(
        "Training dataset: %d rows total (%d base, %d X)",
        len(train_ds), actual_counts["base"], actual_counts["X"],
    )
    return train_ds


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="Path to base YAML config")
    ap.add_argument("--name", required=True, help="Experiment name (overrides config)")
    ap.add_argument("--model-path", required=True, help="Path to pretrained goldfish model")
    ap.add_argument(
        "--data-root",
        required=True,
        help="DATA_ROOT: parent of chunk_XX/ directories",
    )
    ap.add_argument("--chunks", nargs="+", required=True, help="Chunk IDs e.g. 00 01")
    ap.add_argument(
        "--clean-file",
        required=True,
        help=(
            "Path relative to chunk dir (no .txt) for D_base sentences, "
            "e.g. 'Binding-reflexive/train_clean'"
        ),
    )
    ap.add_argument(
        "--matched-file",
        default=None,
        help=(
            "Path relative to chunk dir (no .txt) for D_X sentences, "
            "e.g. 'Binding-reflexive/train_matched'. "
            "All matched sentences are loaded as category='X'. "
            "Omit for the full model (train_full already covers the complete corpus). "
            "For ablated models, include even though these sentences were never trained on — "
            "near-zero influence is the expected signal."
        ),
    )
    ap.add_argument(
        "--phenomena",
        required=True,
        help='JSON dict mapping phenomenon → list of subitem stems, '
             'e.g. \'{"binding": ["principle_A_c_command"]}\'',
    )
    ap.add_argument("--max-base", type=int, default=5_000_000, help="Max D_base sentences; -1 uses all available data")
    ap.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Cap total query pairs via seeded shuffle+select (per phenomenon "
             "when --phenomena has a single key). Omit to use all pairs.",
    )
    ap.add_argument(
        "--full-pairs-parquet",
        default=None,
        help="Path to probe_models.py's --pairs-output parquet for the TRUE "
             "full model (trained on D_base ∪ D_X) on this phenomenon. Combined "
             "with --max-queries and --no-x-pairs-parquet, drives the cascade "
             "sample (see cascade_sample_queries). Alone, drives a simple "
             "learned/not-learned stratified sample. Ignored if --max-queries "
             "is not set.",
    )
    ap.add_argument(
        "--no-x-pairs-parquet",
        default=None,
        help="Path to probe_models.py's --pairs-output parquet for the "
             "ablated (no-X) model under attribution. Combined with "
             "--max-queries and --full-pairs-parquet, drives the cascade "
             "sample: up to 30 'retained' (both models learned it) + up to 30 "
             "'not_learned' (full model never learned it) + the rest from "
             "'learned_specific' (full learned it, this model didn't). Alone "
             "(no --full-pairs-parquet), drives a simple learned/not-learned "
             "stratified sample instead. Ignored if --max-queries is not set.",
    )
    ap.add_argument(
        "--retained-queries", type=int, default=30,
        help="Cascade sample: target count for the 'retained' category (see --no-x-pairs-parquet).",
    )
    ap.add_argument(
        "--not-learned-queries", type=int, default=30,
        help="Cascade sample: target count for the 'not_learned' category (see --no-x-pairs-parquet).",
    )
    ap.add_argument(
        "--artifacts-dir",
        default="artifacts_goldfish",
        help="Artifacts root directory (relative to experiments/)",
    )
    ap.add_argument(
        "--blimp-dir",
        default=str(Path(__file__).resolve().parents[3] / "data" / "blimp"),
        help="Directory for BLiMP .jsonl files (auto-downloaded if missing)",
    )
    ap.add_argument(
        "--manual-npi-tsv",
        default=str(Path(__file__).resolve().parents[3] / "data" / "minimal_pairs_npi.tsv"),
        help="Path to the manually constructed NPI minimal-pair set, used for the "
             "'npi_licensing' phenomenon instead of BLiMP "
             "(see probe_models.py's load_manual_npi_pairs)",
    )
    ap.add_argument("--seed", type=int, default=43, help="Goldfish training seed")
    args = ap.parse_args()

    # Load config and apply overrides
    from src.config import ExperimentCfg

    cfg = ExperimentCfg.loads_yaml(Path(args.config).read_text())
    cfg.name = args.name
    cfg.artifacts_dir = Path(args.artifacts_dir)
    cfg.seeds = [args.seed]

    phenomena: dict[str, list[str]] = json.loads(args.phenomena)
    blimp_dir = Path(args.blimp_dir)
    data_root = Path(args.data_root)
    model_path = Path(args.model_path).resolve()

    logger.info("Experiment : %s", cfg.name)
    logger.info("Artifacts  : %s", cfg.artifacts_dir)
    logger.info("Model path : %s", model_path)
    logger.info("Chunks     : %s", args.chunks)
    logger.info("Clean file : %s", args.clean_file)
    logger.info("Matched    : %s", args.matched_file if args.matched_file else "(none — D_base only)")
    logger.info("Max base   : %d", args.max_base)

    # ── Queries ───────────────────────────────────────────────────────────────
    queries_out = cfg.queries_path()
    if queries_out.exists():
        logger.info("Queries already exist at %s, skipping.", queries_out)
    else:
        logger.info("Preparing queries …")
        ensure_blimp_files(blimp_dir, phenomena)
        queries = build_queries(blimp_dir, phenomena, manual_npi_tsv=Path(args.manual_npi_tsv))
        if args.max_queries is not None and args.max_queries < len(queries):
            if args.full_pairs_parquet and args.no_x_pairs_parquet:
                queries = cascade_sample_queries(
                    queries, args.max_queries,
                    args.full_pairs_parquet, args.no_x_pairs_parquet,
                    seed=args.seed,
                    retained_target=args.retained_queries,
                    not_learned_target=args.not_learned_queries,
                )
            elif args.full_pairs_parquet or args.no_x_pairs_parquet:
                queries = stratified_sample_queries(
                    queries, args.max_queries,
                    args.full_pairs_parquet or args.no_x_pairs_parquet,
                    seed=args.seed,
                )
            else:
                queries = queries.shuffle(seed=args.seed).select(range(args.max_queries))
            logger.info("Subsampled to %d query pairs (seed=%d)", len(queries), args.seed)
        queries_out.mkdir(parents=True, exist_ok=True)
        queries.save_to_disk(str(queries_out))
        logger.info("Saved %d query pairs → %s", len(queries), queries_out)

    # ── Training data ─────────────────────────────────────────────────────────
    train_out = cfg.train_data_path()
    if train_out.exists():
        logger.info("Training data already exists at %s, skipping.", train_out)
    else:
        logger.info("Preparing training data …")
        train_ds = build_training_data(
            data_root, args.chunks, args.clean_file, args.matched_file,
            args.max_base, args.seed,
        )
        train_out.parent.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(train_out))
        logger.info("Saved %d training rows → %s", len(train_ds), train_out)

    # ── Model symlink ─────────────────────────────────────────────────────────
    model_link = cfg.model_dir(args.seed)
    if model_link.exists() or model_link.is_symlink():
        logger.info("Model symlink already exists at %s, skipping.", model_link)
    else:
        model_link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(model_path), str(model_link))
        logger.info("Symlinked %s → %s", model_link, model_path)


if __name__ == "__main__":
    main()
