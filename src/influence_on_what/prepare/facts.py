#!/usr/bin/env python3
"""Prepare query and training datasets for BEAR facts attribution.

Creates HF datasets at the paths expected by `python -m influence_on_what attribute`:
  shared/queries/          — BEAR minimal pairs with fact_key column
  shared/train/            — D_base (clean) + D_X (fact-matched) with fact_ids column
  <tag>/seed_<seed>/model/ — symlink to the pretrained no_bear_facts model

Per-fact matched sentences come from the per-fact files written by the filter step:
  BEAR_FACTS_DIR/chunk_XX/BearFacts/facts/<fact_slug>/cooccurrence/train_matched.txt
  BEAR_FACTS_DIR/chunk_XX/BearFacts/facts/<fact_slug>/subj_occurrence/train_matched.txt (optional)
  BEAR_FACTS_DIR/chunk_XX/BearFacts/facts/<fact_slug>/obj_occurrence/train_matched.txt (optional)

Run once before `python -m influence_on_what attribute`.

Examples
--------
python prepare/facts.py \\
    --config configs/bear_facts_attribution.yaml \\
    --name bear_facts_1b_attribution \\
    --model-path /home/jgsi/InfluenceOnWhat/experiments/artifacts/goldfish_training/models/if_goldfish_1b_no_bear_facts \\
    --bear-facts-dir /home/jgsi/InfluenceOnWhat/data/bear_facts_filter_output_v2 \\
    --chunks 00 01 02 03 04 05 06 07 08 09 10 12 13 14 15 16 \\
    --bear-pairs /home/jgsi/InfluenceOnWhat/experiments/artifacts/check_pretrained/if_goldfish_1b_full/bear_pairs_93716_4.parquet \\
    --corpus-stats /home/jgsi/InfluenceOnWhat/data/bear_corpus_stats.json \\
    --alias-cache /home/jgsi/InfluenceOnWhat/data/wikidata_alias_cache.json \\
    --artifacts-dir /scratch/JOBID
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import random
from pathlib import Path

import datasets
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[prep %(levelname)s %(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Slug helper (must match run_bear_facts_filter.py) ─────────────────────────


def _fact_slug(rel_id: str, subj: str, obj: str) -> str:
    def _safe(s: str) -> str:
        return re.sub(r"[^a-z0-9_]", "", s.lower().replace(" ", "_").replace("-", "_"))
    return f"{rel_id}__{_safe(subj)}__{_safe(obj)}"


# ── Qualifying facts (mirrors run_bear_facts_filter.load_qualifying_pairs) ────


def load_qualifying_triplets(
    pairs_path: Path,
    corpus_stats_path: Path | None,
    corpus_dataset: str,
    min_cooccur_count: int,
    alias_cache_path: Path | None,
    exclude_alias_overlap: bool,
    min_subject_count: int = 10,
    min_object_count: int = 10,
) -> list[tuple[str, str, str]]:
    """Return (subject, correct_object, rel_id) for all qualifying facts.

    Mirrors run_bear_facts_filter.load_qualifying_pairs: a fact qualifies only
    if the model is correct (rank_correct == 1) on EVERY template of the
    relation, and the corpus occurrence thresholds are met. Sourced from
    *pairs_path*, a per-(fact, template) parquet written by
    probe_models.py's --bear-pairs-output (one row per relation/subject/
    correct_object/template_idx with an is_correct column) rather than the
    aggregated correct_facts/wrong_facts lists in the results JSON.
    """
    df = pd.read_parquet(pairs_path, columns=["relation", "subject", "correct_object", "is_correct"])
    learned = df.groupby(["relation", "subject", "correct_object"], sort=False)["is_correct"].all()

    # Build corpus lookup
    corpus_lookup: dict[tuple[str, str], dict] = {}
    if corpus_stats_path and corpus_stats_path.exists():
        with open(corpus_stats_path, encoding="utf-8") as f:
            cs = json.load(f)
        for rel_data in cs.get("relations", {}).values():
            for inst in rel_data.get("instances", []):
                key = (inst["subject"], inst["correct_object"])
                corpus_lookup[key] = inst.get("by_dataset", {}).get(corpus_dataset, {})

    triplets: list[tuple[str, str, str]] = []
    for (rel_id, subject, correct_object), is_learned in learned.items():
        if not is_learned:
            continue
        if corpus_lookup:
            stats = corpus_lookup.get((subject, correct_object), {})
            if (stats.get("subject", 0) < min_subject_count
                    or stats.get("object", 0) < min_object_count
                    or stats.get("cooccur", 0) < min_cooccur_count):
                continue
        triplets.append((subject, correct_object, rel_id))

    logger.info("Qualifying facts before alias-overlap filter: %d", len(triplets))

    if exclude_alias_overlap and alias_cache_path and alias_cache_path.exists():
        with open(alias_cache_path, encoding="utf-8") as f:
            alias_cache: dict[str, list[str]] = json.load(f)

        def _overlap(subj: str, obj: str) -> bool:
            sa = {a.lower() for a in alias_cache.get(subj, [subj])}
            oa = {a.lower() for a in alias_cache.get(obj, [obj])}
            if sa & oa:
                return True
            s, o = subj.lower(), obj.lower()
            if len(s) >= 4 and re.search(r"\b" + re.escape(s) + r"\b", o):
                return True
            if len(o) >= 4 and re.search(r"\b" + re.escape(o) + r"\b", s):
                return True
            return False

        triplets = [(s, o, r) for s, o, r in triplets if not _overlap(s, o)]
        logger.info("Qualifying facts after alias-overlap filter: %d", len(triplets))

    return triplets


# ── Non-learned fact sampling ─────────────────────────────────────────────────


def sample_nonlearned_triplets(
    pairs_path: Path,
    qualifying_set: set[tuple[str, str]],
    max_fully_wrong: int = 30,
    max_one_correct: int = 20,
    max_two_correct: int = 20,
    seed: int = 43,
) -> list[tuple[str, str, str]]:
    """Sample non-qualifying facts by number of correctly-answered templates.

    Buckets facts by their count of is_correct==True rows in *pairs_path* (see
    load_qualifying_triplets for the parquet schema): 0/1/2 correct templates,
    up to max_fully_wrong/max_one_correct/max_two_correct respectively. Facts
    already in qualifying_set are excluded. All caps are upper bounds — fewer
    are returned if the pool is smaller.
    """
    df = pd.read_parquet(pairs_path, columns=["relation", "subject", "correct_object", "is_correct"])
    n_correct = df.groupby(["relation", "subject", "correct_object"], sort=False)["is_correct"].sum()

    by_n_correct: dict[int, list[tuple[str, str, str]]] = {0: [], 1: [], 2: []}
    for (rel_id, subject, correct_object), n in n_correct.items():
        if (subject, correct_object) in qualifying_set:
            continue
        n = int(n)
        if n in by_n_correct:
            by_n_correct[n].append((subject, correct_object, rel_id))

    rng = random.Random(seed)
    result: list[tuple[str, str, str]] = []
    for n, cap in {0: max_fully_wrong, 1: max_one_correct, 2: max_two_correct}.items():
        pool = by_n_correct[n]
        sampled = rng.sample(pool, min(cap, len(pool)))
        logger.info(
            "Non-learned (n_correct=%d): %d available, sampling %d", n, len(pool), len(sampled)
        )
        result.extend(sampled)

    return result


# ── BEAR query building ───────────────────────────────────────────────────────


def _fact_learned_map(pairs_df: pd.DataFrame) -> dict[str, bool]:
    """Return {slug: learned} — learned = is_correct on ALL templates of the fact."""
    learned = pairs_df.groupby(["relation", "subject", "correct_object"], sort=False)["is_correct"].all()
    return {_fact_slug(rel, s, o): bool(v) for (rel, s, o), v in learned.items()}


def _fact_n_correct_map(pairs_df: pd.DataFrame) -> dict[str, int]:
    """Return {slug: n_correct_templates} — count of templates where is_correct."""
    n_correct = pairs_df.groupby(["relation", "subject", "correct_object"], sort=False)["is_correct"].sum()
    return {_fact_slug(rel, s, o): int(v) for (rel, s, o), v in n_correct.items()}


def _fact_avg_scores_map(pairs_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return {slug: (avg_p_correct, avg_entropy_norm)} averaged over all templates of a fact."""
    avg = pairs_df.groupby(["relation", "subject", "correct_object"], sort=False)[
        ["p_correct", "entropy_norm"]
    ].mean()
    return {
        _fact_slug(rel, s, o): (row.p_correct, row.entropy_norm)
        for (rel, s, o), row in avg.iterrows()
    }


def _per_template_scores_from_pairs(
    pairs_df: pd.DataFrame,
) -> dict[tuple[str, int], tuple[float, float]]:
    """Return {(slug, template_idx): (p_correct, entropy_norm)} from the bear-pairs parquet.

    Unlike the aggregated correct_facts/wrong_facts JSON, *pairs_df* has one row
    per (fact, template) with an explicit template_idx, so every template's
    score is available directly — no reconstruction/ambiguity.
    """
    result: dict[tuple[str, int], tuple[float, float]] = {}
    for row in pairs_df.itertuples(index=False):
        slug = _fact_slug(row.relation, row.subject, row.correct_object)
        result[(slug, int(row.template_idx))] = (row.p_correct, row.entropy_norm)
    return result


def build_bear_queries(
    triplets: list[tuple[str, str, str]],
    nonlearned_triplets: list[tuple[str, str, str]] | None = None,
    max_wrong_answers: int = 3,
    full_pairs_df: pd.DataFrame | None = None,
    nbf_pairs_df: pd.DataFrame | None = None,
) -> datasets.Dataset:
    """Build minimal-pair queries from BEAR templates for qualifying and non-learned facts.

    For each fact × template × wrong-answer (up to *max_wrong_answers*), produces
    one (text_good, text_bad) pair.

    Extra columns written to the dataset:
      subject, correct_object    — entity labels
      template_idx               — 0-based index within the relation's templates
      negative_idx               — 0-based index within the chosen wrong answers
      full_n_correct_templates   — number of templates where full model ranks correct answer first
      full_p_correct             — p_correct for this specific template (full model)
      full_entropy_norm          — entropy_norm for this specific template (full model)
      filtered_n_correct_templates    — number of templates where filtered model ranks correct answer
                                        first (-1 when filtered model results unavailable)
      filtered_p_correct              — p_correct for this specific template (filtered model)
      filtered_entropy_norm           — entropy_norm for this specific template (filtered model)
      verdict                         — 'Forgotten' | 'Retained' | 'unknown'
    """
    from lm_pub_quiz import Dataset as BearDataset

    qualifying_set = {(subj, obj) for subj, obj, _ in triplets}
    all_facts_set = qualifying_set | {(subj, obj) for subj, obj, _ in (nonlearned_triplets or [])}

    full_template_scores  = _per_template_scores_from_pairs(full_pairs_df) if full_pairs_df is not None else {}
    full_n_correct        = _fact_n_correct_map(full_pairs_df) if full_pairs_df is not None else {}
    full_avg_scores       = _fact_avg_scores_map(full_pairs_df) if full_pairs_df is not None else {}
    nbf_learned           = _fact_learned_map(nbf_pairs_df) if nbf_pairs_df is not None else {}
    nbf_n_correct         = _fact_n_correct_map(nbf_pairs_df) if nbf_pairs_df is not None else {}
    nbf_template_scores   = _per_template_scores_from_pairs(nbf_pairs_df) if nbf_pairs_df is not None else {}
    nbf_avg_scores        = _fact_avg_scores_map(nbf_pairs_df) if nbf_pairs_df is not None else {}

    (
        good_texts, bad_texts, features, fact_keys,
        subjects, correct_objects,
        template_idxs, negative_idxs,
        full_n_correct_col, full_avg_pc_col, full_avg_ent_col, full_pc_col, full_ent_col,
        filt_n_correct_col, filt_avg_pc_col, filt_avg_ent_col, filt_pc_col, filt_ent_col,
        verdict_col,
    ) = ([] for _ in range(19))
    n_facts_found = 0

    bear = BearDataset.from_name("BEAR")
    for relation in bear:
        rel_id = relation.relation_code
        templates = relation.templates
        answer_space = list(relation.answer_space)
        instance_table = relation.instance_table

        for _, row in instance_table.iterrows():
            subj = str(row["sub_label"])
            correct_idx = int(row["answer_idx"])
            correct_obj = answer_space[correct_idx]

            if (subj, correct_obj) not in all_facts_set:
                continue

            distractors = [
                ans for i, ans in enumerate(answer_space) if i != correct_idx
            ][:max_wrong_answers]
            if not distractors:
                continue

            fkey = _fact_slug(rel_id, subj, correct_obj)
            n_facts_found += 1

            full_learned = (subj, correct_obj) in qualifying_set
            filt_learned = nbf_learned.get(fkey, None)
            filt_n       = nbf_n_correct.get(fkey, -1)
            if filt_learned is None or filt_n == -1:
                verdict = "unknown"
            elif not full_learned:
                verdict = "not_learned"
            elif filt_n == 0:
                verdict = "Forgotten"
            else:
                verdict = "Retained"

            for t_idx, template in enumerate(templates):
                full_pc, full_ent = full_template_scores.get((fkey, t_idx), (float("nan"), float("nan")))
                filt_pc, filt_ent = nbf_template_scores.get((fkey, t_idx), (float("nan"), float("nan")))
                good_text = template.replace("[X]", subj).replace("[Y]", correct_obj)
                for n_idx, distractor in enumerate(distractors):
                    good_texts.append(good_text)
                    bad_texts.append(template.replace("[X]", subj).replace("[Y]", distractor))
                    features.append(rel_id)
                    fact_keys.append(fkey)
                    subjects.append(subj)
                    correct_objects.append(correct_obj)
                    template_idxs.append(t_idx)
                    negative_idxs.append(n_idx)
                    full_avg_pc, full_avg_ent = full_avg_scores.get(fkey, (float("nan"), float("nan")))
                    filt_avg_pc, filt_avg_ent = nbf_avg_scores.get(fkey, (float("nan"), float("nan")))
                    full_n_correct_col.append(full_n_correct.get(fkey, -1))
                    full_avg_pc_col.append(full_avg_pc)
                    full_avg_ent_col.append(full_avg_ent)
                    full_pc_col.append(full_pc)
                    full_ent_col.append(full_ent)
                    filt_n_correct_col.append(nbf_n_correct.get(fkey, -1))
                    filt_avg_pc_col.append(filt_avg_pc)
                    filt_avg_ent_col.append(filt_avg_ent)
                    filt_pc_col.append(filt_pc)
                    filt_ent_col.append(filt_ent)
                    verdict_col.append(verdict)

    logger.info(
        "Built %d query pairs for %d facts (%d qualifying + %d non-learned) "
        "(%d templates × up to %d negatives per fact)",
        len(good_texts), n_facts_found, len(triplets),
        len(nonlearned_triplets or []),
        len(templates) if n_facts_found else 0, max_wrong_answers,
    )
    return datasets.Dataset.from_dict({
        "text_good":              good_texts,
        "text_bad":               bad_texts,
        "feature":                features,
        "fact_key":               fact_keys,
        "subject":                subjects,
        "correct_object":         correct_objects,
        "template_idx":           template_idxs,
        "negative_idx":           negative_idxs,
        "full_n_correct_templates":        full_n_correct_col,
        "full_avg_p_correct":              full_avg_pc_col,
        "full_avg_entropy_norm":           full_avg_ent_col,
        "full_p_correct":                  full_pc_col,
        "full_entropy_norm":               full_ent_col,
        "filtered_n_correct_templates":    filt_n_correct_col,
        "filtered_avg_p_correct":          filt_avg_pc_col,
        "filtered_avg_entropy_norm":       filt_avg_ent_col,
        "filtered_p_correct":              filt_pc_col,
        "filtered_entropy_norm":           filt_ent_col,
        "verdict":                verdict_col,
    })


# ── Cascade BEAR queries (per-template, anchored to the full model) ──────────
# Two groups, each sampled independently (not a shared budget), gated only by
# per-template correctness (not the "correct on every template" fact-level
# criterion used by load_qualifying_triplets) and the corpus co-occurrence
# filter. Both groups' contrast text and probabilities are anchored to the
# TRUE full model's per-template predictions — the no-x (ablated/attributed)
# model's per-template is_correct is used only as the gate, never for text
# or probability values.


def _load_corpus_lookup(
    corpus_stats_path: Path | None, corpus_dataset: str
) -> dict[tuple[str, str], dict]:
    """(subject, correct_object) -> corpus occurrence/co-occurrence counts for
    *corpus_dataset*, from a bear_corpus_stats*.json file. Empty dict (= no
    filtering) if no path given."""
    corpus_lookup: dict[tuple[str, str], dict] = {}
    if corpus_stats_path and corpus_stats_path.exists():
        with open(corpus_stats_path, encoding="utf-8") as f:
            cs = json.load(f)
        for rel_data in cs.get("relations", {}).values():
            for inst in rel_data.get("instances", []):
                key = (inst["subject"], inst["correct_object"])
                corpus_lookup[key] = inst.get("by_dataset", {}).get(corpus_dataset, {})
    return corpus_lookup


def _passes_corpus_filter(
    corpus_lookup: dict[tuple[str, str], dict],
    subject: str,
    correct_object: str,
    min_subject_count: int,
    min_object_count: int,
    min_cooccur_count: int,
) -> bool:
    if not corpus_lookup:
        return True
    stats = corpus_lookup.get((subject, correct_object), {})
    return not (
        stats.get("subject", 0) < min_subject_count
        or stats.get("object", 0) < min_object_count
        or stats.get("cooccur", 0) < min_cooccur_count
    )


def build_cascade_bear_queries(
    full_pairs_df: pd.DataFrame,
    nbf_pairs_df: pd.DataFrame,
    corpus_stats_path: Path | None,
    corpus_dataset: str,
    min_subject_count: int,
    min_object_count: int,
    min_cooccur_count: int,
    learned_specific_target: int,
    not_learned_both_target: int,
    seed: int,
) -> tuple[datasets.Dataset, list[tuple[str, str, str]]]:
    """Build two independently-sampled groups of (fact, template) minimal-pair
    queries:

      learned_specific  — full model correct on this template, no-x model is
                          not. text_good = the correct answer (full model's
                          p_correct); text_bad = the full model's strongest
                          non-gold competitor, i.e. best_other_object at
                          p_incorrect — the "2nd most likely answer", since
                          the model's actual #1 pick here is already correct.

      not_learned_both  — neither model correct on this template. text_good =
                          the actual correct answer (at p_correct); text_bad =
                          the full model's own top-ranked (wrong) answer
                          (predicted_object at p_predicted) — the wrong answer
                          the model actually confuses for the truth here.
                          Kept consistent with learned_specific's good=correct/
                          bad=wrong convention, so ΔI = I(good) - I(bad) means
                          the same thing (pull toward the truth, away from a
                          specific wrong answer) in both groups.

    Up to *learned_specific_target*/*not_learned_both_target* (fact, template)
    rows are sampled (seeded, independent pools) from each group — fewer if a
    pool is smaller. A fact must pass the corpus co-occurrence filter (same
    thresholds/semantics as load_qualifying_triplets) to be eligible for
    either group.

    Returns (queries_dataset, triplets), where triplets is the deduplicated
    (subject, correct_object, rel_id) list across both groups' selected rows,
    for build_training_data's D_X lookup.
    """
    from lm_pub_quiz import Dataset as BearDataset

    corpus_lookup = _load_corpus_lookup(corpus_stats_path, corpus_dataset)

    keep_cols = [
        "relation", "subject", "correct_object", "template_idx", "is_correct",
        "p_correct", "predicted_object", "p_predicted",
        "best_other_object", "p_incorrect",
    ]
    full_df = full_pairs_df[keep_cols].copy()
    nbf_df = nbf_pairs_df[
        ["relation", "subject", "correct_object", "template_idx", "is_correct"]
    ].rename(columns={"is_correct": "nbf_is_correct"})

    merged = full_df.merge(
        nbf_df, on=["relation", "subject", "correct_object", "template_idx"], how="inner"
    )
    logger.info("Cascade: %d (fact, template) rows with both full and no-x labels", len(merged))

    corpus_ok = merged.apply(
        lambda row: _passes_corpus_filter(
            corpus_lookup, row.subject, row.correct_object,
            min_subject_count, min_object_count, min_cooccur_count,
        ),
        axis=1,
    )
    merged = merged[corpus_ok]
    logger.info("Cascade: %d rows pass corpus co-occurrence filter (>= %d)", len(merged), min_cooccur_count)

    learned_specific_pool = merged[merged["is_correct"] & ~merged["nbf_is_correct"]]
    not_learned_both_pool = merged[~merged["is_correct"] & ~merged["nbf_is_correct"]]

    rng = random.Random(seed)

    def _sample(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        idx = list(pool.index)
        rng.shuffle(idx)
        return pool.loc[idx[:n]]

    ls_sample = _sample(learned_specific_pool, learned_specific_target)
    nlb_sample = _sample(not_learned_both_pool, not_learned_both_target)
    logger.info(
        "learned_specific: %d available, sampling %d", len(learned_specific_pool), len(ls_sample)
    )
    logger.info(
        "not_learned_both: %d available, sampling %d", len(not_learned_both_pool), len(nlb_sample)
    )

    bear = BearDataset.from_name("BEAR")
    templates_by_rel = {rel.relation_code: list(rel.templates) for rel in bear}

    (
        good_texts, bad_texts, features, fact_keys,
        subjects, correct_objects, template_idxs, groups,
        p_good_col, p_bad_col, full_p_correct_col,
    ) = ([] for _ in range(11))
    triplets_set: set[tuple[str, str, str]] = set()

    def _emit(row, group: str, good_obj: str, p_good: float, bad_obj: str, p_bad: float) -> None:
        tmpl_list = templates_by_rel.get(row.relation)
        t_idx = int(row.template_idx)
        if tmpl_list is None or t_idx >= len(tmpl_list):
            logger.warning("Skipping %s/%s: template_idx %d out of range for relation %s",
                           row.subject, row.correct_object, t_idx, row.relation)
            return
        template = tmpl_list[t_idx]
        subj = row.subject
        good_texts.append(template.replace("[X]", subj).replace("[Y]", good_obj))
        bad_texts.append(template.replace("[X]", subj).replace("[Y]", bad_obj))
        features.append(row.relation)
        fact_keys.append(_fact_slug(row.relation, subj, row.correct_object))
        subjects.append(subj)
        correct_objects.append(row.correct_object)
        template_idxs.append(t_idx)
        groups.append(group)
        p_good_col.append(p_good)
        p_bad_col.append(p_bad)
        full_p_correct_col.append(row.p_correct)
        triplets_set.add((subj, row.correct_object, row.relation))

    n_skipped_no_competitor = 0
    for row in ls_sample.itertuples(index=False):
        if not row.best_other_object:
            n_skipped_no_competitor += 1
            continue
        _emit(row, "learned_specific", row.correct_object, row.p_correct,
              row.best_other_object, row.p_incorrect)
    if n_skipped_no_competitor:
        logger.warning(
            "learned_specific: skipped %d rows with no non-gold competitor "
            "(single-candidate relation)", n_skipped_no_competitor,
        )

    for row in nlb_sample.itertuples(index=False):
        _emit(row, "not_learned_both", row.correct_object, row.p_correct,
              row.predicted_object, row.p_predicted)

    logger.info(
        "Built %d cascade query pairs (%d learned_specific + %d not_learned_both)",
        len(good_texts), groups.count("learned_specific"), groups.count("not_learned_both"),
    )

    queries = datasets.Dataset.from_dict({
        "text_good": good_texts,
        "text_bad": bad_texts,
        "feature": features,
        "fact_key": fact_keys,
        "subject": subjects,
        "correct_object": correct_objects,
        "template_idx": template_idxs,
        "group": groups,
        "p_good": p_good_col,
        "p_bad": p_bad_col,
        "full_p_correct": full_p_correct_col,
    })
    return queries, sorted(triplets_set)


# ── Training data building ────────────────────────────────────────────────────


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_training_data(
    bear_facts_dir: Path,
    chunks: list[str],
    triplets: list[tuple[str, str, str]],
    max_base: int,
    seed: int = 0,
) -> datasets.Dataset:
    """Build training dataset with category, fact_ids, and source columns.

    D_base rows: from train_clean.txt (no qualifying fact co-occurrence/occurrence),
                 fact_ids="", source="".
    D_X rows:    from per-fact cooccurrence/, subj_occurrence/, and obj_occurrence/
                 train_matched.txt files, fact_ids=comma-separated slugs,
                 source=comma-separated {"cooccur","subj_occur","obj_occur"} tags.
                 A sentence can carry multiple tags if it matched multiple facts'
                 files of different types.

    D_X sentences that match multiple facts carry all matching slugs.
    D_base is reservoir-sampled to at most max_base lines across chunks.
    """
    # ── D_X: collect per-fact sentences, merge fact_ids/source for shared sentences ──
    text_to_facts: dict[str, set[str]] = {}
    text_to_sources: dict[str, set[str]] = {}
    n_cooccur_files = n_occur_files = n_obj_occur_files = 0
    for subj, obj, rel_id in triplets:
        slug = _fact_slug(rel_id, subj, obj)
        for chunk in chunks:
            fact_dir = bear_facts_dir / f"chunk_{chunk}" / "BearFacts" / "facts" / slug
            cooccur_file = fact_dir / "cooccurrence" / "train_matched.txt"
            for line in _read_lines(cooccur_file):
                text_to_facts.setdefault(line, set()).add(slug)
                text_to_sources.setdefault(line, set()).add("cooccur")
            if cooccur_file.exists():
                n_cooccur_files += 1

            occur_file = fact_dir / "subj_occurrence" / "train_matched.txt"
            for line in _read_lines(occur_file):
                text_to_facts.setdefault(line, set()).add(slug)
                text_to_sources.setdefault(line, set()).add("subj_occur")
            if occur_file.exists():
                n_occur_files += 1

            obj_occur_file = fact_dir / "obj_occurrence" / "train_matched.txt"
            for line in _read_lines(obj_occur_file):
                text_to_facts.setdefault(line, set()).add(slug)
                text_to_sources.setdefault(line, set()).add("obj_occur")
            if obj_occur_file.exists():
                n_obj_occur_files += 1

    dx_texts = list(text_to_facts.keys())
    dx_fact_ids = [",".join(sorted(text_to_facts[t])) for t in dx_texts]
    dx_sources = [",".join(sorted(text_to_sources[t])) for t in dx_texts]
    logger.info(
        "D_X: %d unique sentences from %d cooccurrence + %d subj_occurrence + %d "
        "obj_occurrence fact × chunk files (covering %d facts)",
        len(dx_texts), n_cooccur_files, n_occur_files, n_obj_occur_files, len(triplets),
    )

    # ── D_base: count lines per chunk, reservoir-sample to budget ────────────
    counts: dict[str, int] = {}
    for chunk in chunks:
        src = bear_facts_dir / f"chunk_{chunk}" / "BearFacts" / "train_clean.txt"
        counts[chunk] = sum(1 for l in _read_lines(src) if l)
    total_base = sum(counts.values())
    n_base = min(max_base, total_base)
    logger.info("D_base: %d total lines, sampling %d", total_base, n_base)

    rng = random.Random(seed)
    base_texts: list[str] = []
    for chunk in chunks:
        src = bear_facts_dir / f"chunk_{chunk}" / "BearFacts" / "train_clean.txt"
        lines = _read_lines(src)
        ratio = counts[chunk] / total_base if total_base > 0 else 0.0
        n = min(round(ratio * n_base), len(lines))
        if n > 0:
            base_texts.extend(rng.sample(lines, n))
        logger.info("  chunk_%s: sampled %d / %d D_base", chunk, n, len(lines))

    logger.info(
        "Training dataset: %d total (%d base, %d X)",
        len(base_texts) + len(dx_texts), len(base_texts), len(dx_texts),
    )
    return datasets.Dataset.from_dict({
        "text":     base_texts + dx_texts,
        "category": ["base"] * len(base_texts) + ["X"] * len(dx_texts),
        "fact_ids": [""] * len(base_texts) + dx_fact_ids,
        "source":   [""] * len(base_texts) + dx_sources,
    })


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="Path to base YAML config")
    ap.add_argument("--name", required=True, help="Experiment name (overrides config)")
    ap.add_argument("--model-path", required=True, help="Path to no_bear_facts model")
    ap.add_argument("--bear-facts-dir", required=True, help="Root dir of filter outputs (contains chunk_XX/)")
    ap.add_argument("--chunks", nargs="+", required=True, help="Chunk IDs e.g. 00 01 ...")
    ap.add_argument("--bear-pairs", required=True,
                    help="Per-(fact, template) BEAR results parquet for the full model, "
                         "written by probe_models.py's --bear-pairs-output.")
    ap.add_argument("--corpus-stats", default=None, help="bear_corpus_stats.json")
    ap.add_argument("--corpus-dataset", default="1b", help="Dataset key in corpus stats")
    ap.add_argument("--alias-cache", default=None, help="Wikidata alias cache JSON")
    ap.add_argument("--min-subject-count", type=int, default=10)
    ap.add_argument("--min-object-count", type=int, default=10)
    ap.add_argument("--min-cooccur-count", type=int, default=10)
    ap.add_argument("--max-wrong-answers", type=int, default=3,
                    help="Max wrong answers (distractors) per template. "
                         "Each produces one (text_good, text_bad) pair.")
    ap.add_argument("--nbf-bear-pairs", default=None,
                    help="Per-(fact, template) BEAR results parquet for the filtered "
                         "(no-bear-facts) model, written by probe_models.py's "
                         "--bear-pairs-output. Used to populate filtered_fact_learned / "
                         "verdict columns in the queries dataset.")
    ap.add_argument("--max-nonlearned-fully-wrong", type=int, default=30,
                    help="Max non-learned facts with 0 correct templates added to queries.")
    ap.add_argument("--max-nonlearned-one-correct", type=int, default=20,
                    help="Max non-learned facts with exactly 1 correct template added to queries.")
    ap.add_argument("--max-nonlearned-two-correct", type=int, default=20,
                    help="Max non-learned facts with exactly 2 correct templates added to queries.")
    ap.add_argument("--max-base", type=int, default=5_000_000, help="Max D_base sentences")
    ap.add_argument("--artifacts-dir", default="artifacts_attribution")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument(
        "--learned-specific-queries", type=int, default=None,
        help="Switch to cascade query mode (see build_cascade_bear_queries): "
             "sample up to this many (fact, template) rows where the full "
             "model got it right and the no-x model didn't. Requires "
             "--not-learned-both-queries to also be set. Bypasses the "
             "default fact-level (all-templates-correct) query builder "
             "entirely, including --max-wrong-answers.",
    )
    ap.add_argument(
        "--not-learned-both-queries", type=int, default=None,
        help="Cascade query mode: sample up to this many (fact, template) "
             "rows where neither model got it right. Requires "
             "--learned-specific-queries to also be set.",
    )
    args = ap.parse_args()

    from ..config import ExperimentCfg

    cfg = ExperimentCfg.loads_yaml(Path(args.config).read_text())
    cfg.name = args.name
    cfg.artifacts_dir = Path(args.artifacts_dir)
    cfg.seeds = [args.seed]

    bear_facts_dir = Path(args.bear_facts_dir)
    model_path = Path(args.model_path).resolve()

    logger.info("Experiment : %s", cfg.name)
    logger.info("Artifacts  : %s", cfg.artifacts_dir)
    logger.info("Model      : %s", model_path)
    logger.info("Bear facts : %s", bear_facts_dir)
    logger.info("Chunks     : %s", args.chunks)

    cascade_mode = args.learned_specific_queries is not None or args.not_learned_both_queries is not None
    if cascade_mode and (args.learned_specific_queries is None or args.not_learned_both_queries is None):
        raise RuntimeError(
            "--learned-specific-queries and --not-learned-both-queries must be set together."
        )

    if cascade_mode:
        # ── Cascade queries (per-template, anchored to the full model) ───────
        if not args.nbf_bear_pairs:
            raise RuntimeError("Cascade mode requires --nbf-bear-pairs (no-x model's bear-pairs).")
        full_pairs_df = pd.read_parquet(args.bear_pairs)
        nbf_pairs_df = pd.read_parquet(args.nbf_bear_pairs)
        logger.info("Loaded full model bear-pairs from %s", args.bear_pairs)
        logger.info("Loaded no-x model bear-pairs from %s", args.nbf_bear_pairs)

        queries, triplets = build_cascade_bear_queries(
            full_pairs_df=full_pairs_df,
            nbf_pairs_df=nbf_pairs_df,
            corpus_stats_path=Path(args.corpus_stats) if args.corpus_stats else None,
            corpus_dataset=args.corpus_dataset,
            min_subject_count=args.min_subject_count,
            min_object_count=args.min_object_count,
            min_cooccur_count=args.min_cooccur_count,
            learned_specific_target=args.learned_specific_queries,
            not_learned_both_target=args.not_learned_both_queries,
            seed=args.seed,
        )
        if not triplets:
            raise RuntimeError("No cascade query pairs found — check filter arguments.")

        facts_out = cfg.shared_dir() / "qualifying_facts.json"
        facts_out.parent.mkdir(parents=True, exist_ok=True)
        with open(facts_out, "w") as f:
            json.dump(
                [{"subject": s, "object": o, "rel_id": r} for s, o, r in triplets],
                f, indent=2,
            )
        logger.info("Saved %d facts referenced by cascade queries → %s", len(triplets), facts_out)

        queries_out = cfg.queries_path()
        if queries_out.exists():
            logger.info("Queries already exist at %s, skipping.", queries_out)
        else:
            queries_out.mkdir(parents=True, exist_ok=True)
            queries.save_to_disk(str(queries_out))
            logger.info("Saved %d cascade query pairs → %s", len(queries), queries_out)
    else:
        # ── Qualifying facts ──────────────────────────────────────────────────
        triplets = load_qualifying_triplets(
            pairs_path=Path(args.bear_pairs),
            corpus_stats_path=Path(args.corpus_stats) if args.corpus_stats else None,
            corpus_dataset=args.corpus_dataset,
            min_subject_count=args.min_subject_count,
            min_object_count=args.min_object_count,
            min_cooccur_count=args.min_cooccur_count,
            alias_cache_path=Path(args.alias_cache) if args.alias_cache else None,
            exclude_alias_overlap=True,
        )
        if not triplets:
            raise RuntimeError("No qualifying facts found — check filter arguments.")

        # Save qualifying facts list for reference
        facts_out = cfg.shared_dir() / "qualifying_facts.json"
        facts_out.parent.mkdir(parents=True, exist_ok=True)
        with open(facts_out, "w") as f:
            json.dump(
                [{"subject": s, "object": o, "rel_id": r} for s, o, r in triplets],
                f, indent=2,
            )
        logger.info("Saved %d qualifying facts → %s", len(triplets), facts_out)

        # ── Non-learned fact sample ────────────────────────────────────────────
        qualifying_pairs = {(s, o) for s, o, _ in triplets}
        nonlearned_triplets = sample_nonlearned_triplets(
            pairs_path=Path(args.bear_pairs),
            qualifying_set=qualifying_pairs,
            max_fully_wrong=args.max_nonlearned_fully_wrong,
            max_one_correct=args.max_nonlearned_one_correct,
            max_two_correct=args.max_nonlearned_two_correct,
            seed=args.seed,
        )
        nonlearned_out = cfg.shared_dir() / "nonlearned_facts.json"
        with open(nonlearned_out, "w") as f:
            json.dump(
                [{"subject": s, "object": o, "rel_id": r} for s, o, r in nonlearned_triplets],
                f, indent=2,
            )
        logger.info("Saved %d non-learned facts → %s", len(nonlearned_triplets), nonlearned_out)

        # ── Queries ─────────────────────────────────────────────────────────────
        queries_out = cfg.queries_path()
        if queries_out.exists():
            logger.info("Queries already exist at %s, skipping.", queries_out)
        else:
            full_pairs_df = pd.read_parquet(args.bear_pairs)
            logger.info("Loaded full model bear-pairs from %s", args.bear_pairs)

            nbf_pairs_df = None
            if args.nbf_bear_pairs:
                nbf_pairs_df = pd.read_parquet(args.nbf_bear_pairs)
                logger.info("Loaded NBF bear-pairs from %s", args.nbf_bear_pairs)

            queries = build_bear_queries(
                triplets,
                nonlearned_triplets=nonlearned_triplets,
                max_wrong_answers=args.max_wrong_answers,
                full_pairs_df=full_pairs_df,
                nbf_pairs_df=nbf_pairs_df,
            )

            queries_out.mkdir(parents=True, exist_ok=True)
            queries.save_to_disk(str(queries_out))
            logger.info("Saved %d query pairs → %s", len(queries), queries_out)

    # ── Training data ─────────────────────────────────────────────────────────
    train_out = cfg.train_data_path()
    if train_out.exists():
        logger.info("Training data already exists at %s, skipping.", train_out)
    else:
        train_ds = build_training_data(
            bear_facts_dir, args.chunks, triplets, args.max_base, args.seed,
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
