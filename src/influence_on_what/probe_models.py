#!/usr/bin/env python3
"""Evaluate whether a pretrained (HuggingFace) causal LM has learned key features.

Phenomena tested (BLiMP sub-tasks grouped by category, plus BEAR):
  anaphor_agreement        — gender & number agreement of anaphors
  argument_structure       — passive/transitive/causative/etc.
  binding                  — Principle A (c-command, case, domain, reconstruction)
  control_raising          — existential there raising, expletive it, tough vs raising
  determiner_noun_agreement — det-noun agreement (regular, irregular, with adj)
  ellipsis                 — N-bar ellipsis
  filler_gap               — wh-questions and wh-vs-that
  irregular_forms          — irregular past participle (adj & verb)
  island_effects           — adjunct, complex NP, coordinate, left branch, sentential, wh islands
  npi_licensing            — NPI licensing (matrix Q, npi_present, only, sentential negation)
  quantifiers              — existential there quantifiers, superlative quantifiers
  subject_verb_agreement   — distractor agreement, irregular & regular plural SVA

BEAR probing (run by default, skip with --no-bear):
  Uses lm-pub-quiz to probe all 60 BEAR relations via pseudo-log-likelihood scoring.
  For each relation the model's predicted answer is the candidate with the highest PLL;
  reported metrics per relation and overall:
    accuracy       — fraction of instances where the model's top-1 prediction is correct
    n              — number of instances evaluated
    p_predicted    — softmax probability assigned to the model's predicted answer
    p_predicted_above_chance — p_predicted × answer_space_size (1.0 = chance; comparable across relations)
    p_correct      — softmax probability assigned to the gold answer
    rank_correct   — rank of the gold answer among all candidates (1 = top-1)
    entropy_norm   — normalized entropy over candidate distribution (0 = confident, 1 = uniform)
  Results are stored under the "bear" key with "by_relation" breakdowns and
  "correct_facts" / "wrong_facts" lists of {subject, correct_object, predicted_object, …}.

Metrics follow evaluate.py:
  full_sentence  — log P(s) = Σ log p(tᵢ|t_{<i}); higher = preferred
  length_norm    — log P(s) / n_tokens; higher = preferred
  one_prefix     — log p(diverging token | shared prefix)
  two_prefix     — log p(re-convergence token | good/bad prefix)

Usage:
  python -m influence_on_what.probe_models
  python -m influence_on_what.probe_models --model gpt2
  python -m influence_on_what.probe_models --model openai-community/gpt2-medium
  python -m influence_on_what.probe_models --phenomena npi_licensing --output results.json
  python -m influence_on_what.probe_models --phenomena quantifiers binding  # skip BEAR (omit facts_bear)
  python -m influence_on_what.probe_models --bear-batch-size 8           # faster BEAR with larger batches
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

BLIMP_BASE = "https://raw.githubusercontent.com/alexwarstadt/blimp/master/data"

# ── BLiMP file groupings ──────────────────────────────────────────────────────


BLIMP_FILES_ALL: dict[str, list[str]] = {
    "anaphor_agreement": [
        "anaphor_gender_agreement.jsonl",
        "anaphor_number_agreement.jsonl",
    ],
    "argument_structure": [
        "animate_subject_passive.jsonl",
        "animate_subject_trans.jsonl",
        "causative.jsonl",
        "drop_argument.jsonl",
        "inchoative.jsonl",
        "intransitive.jsonl",
        "passive_1.jsonl",
        "passive_2.jsonl",
        "transitive.jsonl",
    ],
    "binding": [
        "principle_A_c_command.jsonl",
        "principle_A_case_1.jsonl",
        "principle_A_case_2.jsonl",
        "principle_A_domain_1.jsonl",
        "principle_A_domain_2.jsonl",
        "principle_A_domain_3.jsonl",
        "principle_A_reconstruction.jsonl",
    ],
    "control_raising": [
        "existential_there_object_raising.jsonl",
        "existential_there_subject_raising.jsonl",
        "expletive_it_object_raising.jsonl",
        "tough_vs_raising_1.jsonl",
        "tough_vs_raising_2.jsonl",
    ],
    "determiner_noun_agreement": [
        "determiner_noun_agreement_1.jsonl",
        "determiner_noun_agreement_2.jsonl",
        "determiner_noun_agreement_irregular_1.jsonl",
        "determiner_noun_agreement_irregular_2.jsonl",
        "determiner_noun_agreement_with_adjective_1.jsonl",
        "determiner_noun_agreement_with_adj_2.jsonl",
        "determiner_noun_agreement_with_adj_irregular_1.jsonl",
        "determiner_noun_agreement_with_adj_irregular_2.jsonl",
    ],
    "ellipsis": [
        "ellipsis_n_bar_1.jsonl",
        "ellipsis_n_bar_2.jsonl",
    ],
    "filler_gap": [
        "wh_questions_object_gap.jsonl",
        "wh_questions_subject_gap.jsonl",
        "wh_questions_subject_gap_long_distance.jsonl",
        "wh_vs_that_no_gap.jsonl",
        "wh_vs_that_no_gap_long_distance.jsonl",
        "wh_vs_that_with_gap.jsonl",
        "wh_vs_that_with_gap_long_distance.jsonl",
    ],
    "irregular_forms": [
        "irregular_past_participle_adjectives.jsonl",
        "irregular_past_participle_verbs.jsonl",
    ],
    "island_effects": [
        "adjunct_island.jsonl",
        "complex_NP_island.jsonl",
        "coordinate_structure_constraint_complex_left_branch.jsonl",
        "coordinate_structure_constraint_object_extraction.jsonl",
        "left_branch_island_echo_question.jsonl",
        "left_branch_island_simple_question.jsonl",
        "sentential_subject_island.jsonl",
        "wh_island.jsonl",
    ],
    "npi_licensing": [
        "matrix_question_npi_licensor_present.jsonl",
        "npi_present_1.jsonl",
        "npi_present_2.jsonl",
        "only_npi_licensor_present.jsonl",
        "only_npi_scope.jsonl",
        "sentential_negation_npi_licensor_present.jsonl",
        "sentential_negation_npi_scope.jsonl",
    ],
    "quantifiers": [
        "existential_there_quantifiers_1.jsonl",
        "existential_there_quantifiers_2.jsonl",
        "superlative_quantifiers_1.jsonl",
        "superlative_quantifiers_2.jsonl",
    ],
    "subject_verb_agreement": [
        "distractor_agreement_relational_noun.jsonl",
        "distractor_agreement_relative_clause.jsonl",
        "irregular_plural_subject_verb_agreement_1.jsonl",
        "irregular_plural_subject_verb_agreement_2.jsonl",
        "regular_plural_subject_verb_agreement_1.jsonl",
        "regular_plural_subject_verb_agreement_2.jsonl",
    ],
}

BLIMP_FILES_FILTERED: dict[str, list[str]] = {
    "binding": [
        "principle_A_c_command.jsonl",
    ],
    "island_effects": [
        "left_branch_island_simple_question.jsonl",
    ],
    "npi_licensing": [
        "matrix_question_npi_licensor_present.jsonl",
        "npi_present_1.jsonl",
        "npi_present_2.jsonl",
        "only_npi_licensor_present.jsonl",
        "only_npi_scope.jsonl",
        "sentential_negation_npi_licensor_present.jsonl",
        "sentential_negation_npi_scope.jsonl",
    ],
    "quantifiers": [
        "existential_there_quantifiers_1.jsonl",
        "existential_there_quantifiers_2.jsonl",
    ],
}

BLIMP_FILES = BLIMP_FILES_FILTERED  # switch to BLIMP_FILES_ALL to run on all files

ALL_PHENOMENA = [
    "facts_bear",
    "manual_npi",
    "anaphor_agreement",
    "argument_structure",
    "binding",
    "control_raising",
    "determiner_noun_agreement",
    "ellipsis",
    "filler_gap",
    "irregular_forms",
    "island_effects",
    "npi_licensing",
    "quantifiers",
    "subject_verb_agreement",
]

# ── scoring primitives (mirrors experiments/src/evaluate.py) ─────────────────


def _sentence_scores(
    model, tokenizer, text: str, max_length: int, device: str
) -> tuple[float, float]:
    """Return (logp, logp_norm) where logp = Σ log p(tᵢ|t_{<i}) and logp_norm = logp / n_tokens."""
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length, padding=False
    )
    ids = enc.input_ids.to(device)
    n = ids.shape[1]
    with torch.no_grad():
        loss = model(input_ids=ids, labels=ids).loss
    logp = -float(loss.item()) * n
    return logp, logp / n


def _tokenize(tokenizer, text: str, max_length: int) -> torch.Tensor:
    enc = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length, padding=False
    )
    return enc.input_ids[0]


def _divergence_point(a: torch.Tensor, b: torch.Tensor) -> int:
    min_len = min(len(a), len(b))
    for k in range(min_len):
        if a[k] != b[k]:
            return k
    return min_len


def _one_prefix_scores(
    model,
    tokenizer,
    text_good: str,
    text_bad: str,
    max_length: int,
    device: str,
    *,
    prefix_text: str | None = None,
) -> tuple[float, float, bool]:
    ids_g = _tokenize(tokenizer, text_good, max_length)
    ids_b = _tokenize(tokenizer, text_bad, max_length)
    used_blimp = prefix_text is not None
    if used_blimp:
        k = len(_tokenize(tokenizer, prefix_text, max_length))
    else:
        k = _divergence_point(ids_g, ids_b)
    if k == 0 or k >= len(ids_g) or k >= len(ids_b):
        return math.nan, math.nan, used_blimp
    prefix = ids_g[:k].unsqueeze(0).to(device)
    with torch.no_grad():
        log_probs = torch.log_softmax(model(input_ids=prefix).logits[0, -1], dim=-1)
    return log_probs[ids_g[k]].item(), log_probs[ids_b[k]].item(), used_blimp


def _two_prefix_scores(
    model,
    tokenizer,
    text_good: str,
    text_bad: str,
    max_length: int,
    device: str,
    *,
    prefix_good_text: str | None = None,
    prefix_bad_text: str | None = None,
) -> tuple[float, float, bool]:
    ids_g = _tokenize(tokenizer, text_good, max_length)
    ids_b = _tokenize(tokenizer, text_bad, max_length)

    prefix_g = prefix_b = None
    t_star = None
    used_blimp = False

    if prefix_good_text is not None and prefix_bad_text is not None:
        k_g = len(_tokenize(tokenizer, prefix_good_text, max_length))
        k_b = len(_tokenize(tokenizer, prefix_bad_text, max_length))
        if k_g < len(ids_g) and k_b < len(ids_b):
            t_g, t_b = int(ids_g[k_g].item()), int(ids_b[k_b].item())
            if t_g == t_b:
                t_star = t_g
                prefix_g = ids_g[:k_g].unsqueeze(0).to(device)
                prefix_b = ids_b[:k_b].unsqueeze(0).to(device)
                used_blimp = True

    if prefix_g is None:
        k = _divergence_point(ids_g, ids_b)
        if k >= len(ids_g) or k >= len(ids_b):
            return math.nan, math.nan, False
        t_star_pos = None
        for i in range(k + 1, min(len(ids_g), len(ids_b))):
            if ids_g[i] == ids_b[i]:
                t_star_pos = i
                break
        if t_star_pos is None:
            return math.nan, math.nan, False
        t_star = int(ids_g[t_star_pos].item())
        prefix_g = ids_g[:t_star_pos].unsqueeze(0).to(device)
        prefix_b = ids_b[:t_star_pos].unsqueeze(0).to(device)

    with torch.no_grad():
        lp_g = torch.log_softmax(model(input_ids=prefix_g).logits[0, -1], dim=-1)[
            t_star
        ].item()
        lp_b = torch.log_softmax(model(input_ids=prefix_b).logits[0, -1], dim=-1)[
            t_star
        ].item()
    return lp_g, lp_b, used_blimp


def _generation_accuracy(
    model, tokenizer, text_good: str, text_bad: str, max_length: int, device: str
) -> bool | None:
    ids_g = _tokenize(tokenizer, text_good, max_length)
    ids_b = _tokenize(tokenizer, text_bad, max_length)
    k = _divergence_point(ids_g, ids_b)
    if k == 0 or k >= len(ids_g) or k >= len(ids_b):
        return None
    pos = k
    while pos < len(ids_g):
        if pos < len(ids_b) and ids_g[pos] == ids_b[pos]:
            return True
        prefix = ids_g[:pos].unsqueeze(0).to(device)
        with torch.no_grad():
            predicted = int(model(input_ids=prefix).logits[0, -1].argmax().item())
        if predicted != int(ids_g[pos].item()):
            return False
        pos += 1
    return True


# ── data loading ──────────────────────────────────────────────────────────────


def load_blimp_pairs(blimp_dir: Path, phenomenon: str) -> dict[str, list[dict]]:
    """Returns {file_stem: [pairs]} for each BLiMP file in the phenomenon."""
    result: dict[str, list[dict]] = {}
    for fname in BLIMP_FILES[phenomenon]:
        fpath = blimp_dir / fname
        if not fpath.exists():
            logger.warning(f"BLiMP file not found: {fpath}")
            continue
        pairs: list[dict] = []
        with open(fpath) as f:
            for line in f:
                entry = json.loads(line)
                pairs.append(
                    {
                        "text_good": entry["sentence_good"],
                        "text_bad": entry["sentence_bad"],
                        "uid": entry.get("UID", fpath.stem),
                        "one_prefix_method": entry.get("one_prefix_method", False),
                        "two_prefix_method": entry.get("two_prefix_method", False),
                        "one_prefix_prefix": entry.get("one_prefix_prefix"),
                        "two_prefix_prefix_good": entry.get("two_prefix_prefix_good"),
                        "two_prefix_prefix_bad": entry.get("two_prefix_prefix_bad"),
                    }
                )
        if pairs:
            result[fpath.stem] = pairs
    return result


def load_manual_npi_pairs(tsv_path: Path) -> list[dict]:
    """Returns a flat list of pairs with full metadata from the TSV."""
    pairs: list[dict] = []
    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pairs.append(
                {
                    "text_good": row["sen"],
                    "text_bad": row["wrong_sen"],
                    "uid": row.get("sent_id", ""),
                    "lemma": row.get("lemma", ""),
                    "env": row.get("env", ""),
                    "npi_to_neg_distance": row.get("npi_to_neg_distance", ""),
                    "sent_len": row.get("sent_len", ""),
                    "multiple_npis": row.get("multiple_npis", ""),
                }
            )
    return pairs


# ── evaluation ────────────────────────────────────────────────────────────────

_METRIC_LABELS = [
    "full_sentence",
    "softmax_prob",
    "length_norm",
    "one_prefix",
    "two_prefix",
]

# A pair counts as "learned" when the softmax probability of the good sentence
# (over the 2-way {good, bad} full-sentence logprob softmax) exceeds this.
SOFTMAX_PROB_LEARNED_THRESHOLD = 0.5


def _softmax_prob_good(score_good: float, score_bad: float) -> float:
    """P(good) under a 2-way softmax over (score_good, score_bad)."""
    max_s = max(score_good, score_bad)
    eg = math.exp(score_good - max_s)
    eb = math.exp(score_bad - max_s)
    return eg / (eg + eb)


def _binary_entropy(score_good: float, score_bad: float) -> float | None:
    """Binary entropy of 2-way softmax over (score_good, score_bad). None if either is nan."""
    if math.isnan(score_good) or math.isnan(score_bad):
        return None
    max_s = max(score_good, score_bad)
    eg = math.exp(score_good - max_s)
    eb = math.exp(score_bad - max_s)
    total = eg + eb
    pg, pb = eg / total, eb / total
    h = 0.0
    if pg > 0:
        h -= pg * math.log(pg)
    if pb > 0:
        h -= pb * math.log(pb)
    return h


def _mean(xs: list) -> float | None:
    valid = [x for x in xs if x is not None]
    return sum(valid) / len(valid) if valid else None


def evaluate_pairs(
    model,
    tokenizer,
    pairs: list[dict],
    max_length: int,
    device: str,
    collect_per_pair: bool = False,
    log_interval: int = 50,
) -> dict:
    fs_hits: list[bool] = []
    fs_scores: list[tuple[float, float]] = []
    sp_probs: list[float] = []
    ln_hits: list[bool] = []
    ln_scores: list[tuple[float, float]] = []
    op_blimp_hits: list[bool] = []
    op_blimp_scores: list[tuple[float, float]] = []
    op_fallback_hits: list[bool] = []
    op_fallback_scores: list[tuple[float, float]] = []
    tp_blimp_hits: list[bool] = []
    tp_blimp_scores: list[tuple[float, float]] = []
    tp_fallback_hits: list[bool] = []
    tp_fallback_scores: list[tuple[float, float]] = []
    per_pair_results: list[dict] = []

    for i, p in enumerate(pairs):
        good, bad = p["text_good"], p["text_bad"]

        lp_g, ln_g = _sentence_scores(model, tokenizer, good, max_length, device)
        lp_b, ln_b = _sentence_scores(model, tokenizer, bad, max_length, device)
        sp_g = _softmax_prob_good(lp_g, lp_b)
        fs_hits.append(lp_g > lp_b)
        fs_scores.append((lp_g, lp_b))
        sp_probs.append(sp_g)
        ln_hits.append(ln_g > ln_b)
        ln_scores.append((ln_g, ln_b))

        op_g, op_b, op_blimp = _one_prefix_scores(
            model,
            tokenizer,
            good,
            bad,
            max_length,
            device,
            prefix_text=p.get("one_prefix_prefix"),
        )
        if not math.isnan(op_g):
            if op_blimp:
                op_blimp_hits.append(op_g > op_b)
                op_blimp_scores.append((op_g, op_b))
            else:
                op_fallback_hits.append(op_g > op_b)
                op_fallback_scores.append((op_g, op_b))

        tp_g, tp_b, tp_blimp = _two_prefix_scores(
            model,
            tokenizer,
            good,
            bad,
            max_length,
            device,
            prefix_good_text=p.get("two_prefix_prefix_good"),
            prefix_bad_text=p.get("two_prefix_prefix_bad"),
        )
        if not math.isnan(tp_g):
            if tp_blimp:
                tp_blimp_hits.append(tp_g > tp_b)
                tp_blimp_scores.append((tp_g, tp_b))
            else:
                tp_fallback_hits.append(tp_g > tp_b)
                tp_fallback_scores.append((tp_g, tp_b))

        if collect_per_pair:
            per_pair_results.append(
                {
                    "pair_idx": i,
                    "sent_id": p.get("uid", ""),
                    "env": p.get("env", ""),
                    "lemma": p.get("lemma", ""),
                    "npi_to_neg_distance": p.get("npi_to_neg_distance", ""),
                    "sent_len": p.get("sent_len", ""),
                    "multiple_npis": p.get("multiple_npis", ""),
                    "text_good": good,
                    "text_bad": bad,
                    "logp_good": lp_g,
                    "logp_bad": lp_b,
                    "logp_norm_good": ln_g,
                    "logp_norm_bad": ln_b,
                    "op_logp_good": None if math.isnan(op_g) else op_g,
                    "op_logp_bad": None if math.isnan(op_b) else op_b,
                    "tp_logp_good": None if math.isnan(tp_g) else tp_g,
                    "tp_logp_bad": None if math.isnan(tp_b) else tp_b,
                    "entropy_full_sentence": _binary_entropy(lp_g, lp_b),
                    "entropy_length_norm": _binary_entropy(ln_g, ln_b),
                    "entropy_one_prefix": _binary_entropy(op_g, op_b),
                    "entropy_two_prefix": _binary_entropy(tp_g, tp_b),
                    "full_sentence_correct": bool(lp_g > lp_b),
                    "softmax_prob_good": sp_g,
                    "learned_x": bool(sp_g > SOFTMAX_PROB_LEARNED_THRESHOLD),
                    "length_norm_correct": bool(ln_g > ln_b),
                    "one_prefix_correct": bool(op_g > op_b)
                    if not math.isnan(op_g)
                    else None,
                    "two_prefix_correct": bool(tp_g > tp_b)
                    if not math.isnan(tp_g)
                    else None,
                }
            )

        if (i + 1) % log_interval == 0:
            logger.info(f"  {i + 1}/{len(pairs)} pairs done")

    def _stat(
        hits: list[bool], scores: list[tuple[float, float]] | None = None
    ) -> dict | None:
        if not hits:
            return None
        result: dict = {"acc": sum(hits) / len(hits), "n": len(hits)}
        if scores:
            result["mean_good"] = sum(s[0] for s in scores) / len(scores)
            result["mean_bad"] = sum(s[1] for s in scores) / len(scores)
            result["mean_entropy"] = _mean([_binary_entropy(g, b) for g, b in scores])
        return result

    def _stat_prob(probs: list[float], threshold: float) -> dict | None:
        """acc = fraction of pairs with P(good) > threshold ('learned')."""
        if not probs:
            return None
        return {
            "acc": sum(p > threshold for p in probs) / len(probs),
            "n": len(probs),
            "threshold": threshold,
            "mean_p_good": sum(probs) / len(probs),
        }

    def _stat_split(
        blimp: list[bool],
        fallback: list[bool],
        blimp_scores: list[tuple[float, float]] | None = None,
        fallback_scores: list[tuple[float, float]] | None = None,
    ) -> dict | None:
        all_hits = blimp + fallback
        if not all_hits:
            return None
        result: dict = {
            "acc": sum(all_hits) / len(all_hits),
            "n": len(all_hits),
            "blimp_n": len(blimp),
            "fallback_n": len(fallback),
        }
        all_scores = (blimp_scores or []) + (fallback_scores or [])
        if all_scores:
            result["mean_good"] = sum(s[0] for s in all_scores) / len(all_scores)
            result["mean_bad"] = sum(s[1] for s in all_scores) / len(all_scores)
            result["mean_entropy"] = _mean(
                [_binary_entropy(g, b) for g, b in all_scores]
            )
        return result

    result: dict = {
        "n_pairs": len(pairs),
        "full_sentence": _stat(fs_hits, fs_scores or None),
        "softmax_prob": _stat_prob(sp_probs, SOFTMAX_PROB_LEARNED_THRESHOLD),
        "length_norm": _stat(ln_hits, ln_scores or None),
        "one_prefix": _stat_split(
            op_blimp_hits,
            op_fallback_hits,
            op_blimp_scores or None,
            op_fallback_scores or None,
        ),
        "two_prefix": _stat_split(
            tp_blimp_hits,
            tp_fallback_hits,
            tp_blimp_scores or None,
            tp_fallback_scores or None,
        ),
    }
    if collect_per_pair:
        result["items"] = per_pair_results
    return result


# ── BEAR probing ──────────────────────────────────────────────────────────────


def run_bear_probing(
    model: PreTrainedModel,
    tokenizer,
    device: str,
    model_name: str,
    batch_size: int = 1,
) -> dict:
    """Run BEAR probing with lm-pub-quiz on all 60 relations.

    Uses the already-loaded model so no second load is needed.  Candidates are
    ranked by their summed pseudo-log-likelihood (PLL); the length-normalized
    (per-token mean) PLL is computed in the same pass and reported alongside it.
    Softmax probabilities over the candidate set (p_correct, p_predicted,
    entropy_norm) are stored for interpretability.
    """
    from lm_pub_quiz import Dataset
    from lm_pub_quiz.evaluators.pll_evaluators import CausalLMEvaluator

    dataset = Dataset.from_name("BEAR")
    evaluator = CausalLMEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device),
        model_name=model_name,
    )
    # reduction=None returns per-token PLL scores so we can derive both the
    # summed PLL (the default metric) and the length-normalized (per-token mean)
    # PLL in a single pass.
    dataset_results = evaluator.evaluate_dataset(
        dataset, batch_size=batch_size, reduction=None
    )

    by_relation: dict[str, dict] = {}
    total_correct = 0
    total_n = 0
    bear_items: list[dict] = []

    for rel_result in dataset_results:
        rel_code = rel_result.relation_code
        table = rel_result.instance_table
        answer_space = (
            rel_result.answer_space
        )  # pd.Series: index=wikidata_id, values=labels

        # Each cell is a list of per-token PLL scores (reduction=None). Stack
        # into (N, K) matrices: raw = summed PLL (the default metric);
        # lennorm = per-token mean (length-normalized) over the same tokens.
        raw_matrix = np.stack(
            [
                np.array([sum(c) for c in r], dtype=np.float64)
                for r in table["pll_scores"].values
            ]
        )
        lennorm_matrix = np.stack(
            [
                np.array([sum(c) / max(len(c), 1) for c in r], dtype=np.float64)
                for r in table["pll_scores"].values
            ]
        )

        correct_facts: list[dict] = []
        wrong_facts: list[dict] = []

        # Collect per-template results keyed by instance so we can apply the
        # all-templates-correct criterion at the subject level.
        instance_rows: dict[int, list[dict]] = {}
        for row_idx, (_, row) in enumerate(table.iterrows()):
            raw = raw_matrix[row_idx]
            predicted_idx = int(np.argmax(raw))
            correct_idx = int(row["answer_idx"])
            is_correct = predicted_idx == correct_idx

            exp_s = np.exp(raw - raw.max())
            probs = exp_s / exp_s.sum()
            n_cands = len(probs)
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
            entropy_norm = entropy / np.log(n_cands) if n_cands > 1 else 0.0
            rank_correct = int(np.sum(raw > raw[correct_idx])) + 1

            # Length-normalized (per-token mean PLL) softmax over candidates.
            ln = lennorm_matrix[row_idx]
            ln_exp = np.exp(ln - ln.max())
            ln_probs = ln_exp / ln_exp.sum()
            ln_predicted_idx = int(np.argmax(ln))

            # Log-PLL margin of the gold answer over its strongest distractor
            # (the factual analog of the linguistic summed-log-prob delta
            # logP(s+) - logP(s-)). >0 iff gold is top-1. Both the summed and the
            # length-normalized variants are stored.
            _other_mask = np.arange(len(raw)) != correct_idx
            if _other_mask.any():
                _best_other = int(np.flatnonzero(_other_mask)[np.argmax(raw[_other_mask])])
                margin = float(raw[correct_idx] - raw[_best_other])
                p_incorrect = float(probs[_best_other])
                best_other_object = str(answer_space.iloc[_best_other])
                _ln_best_other = int(np.flatnonzero(_other_mask)[np.argmax(ln[_other_mask])])
                margin_lennorm = float(ln[correct_idx] - ln[_ln_best_other])
                p_incorrect_lennorm = float(ln_probs[_ln_best_other])
            else:
                margin = margin_lennorm = float("inf")
                p_incorrect = p_incorrect_lennorm = 0.0
                best_other_object = ""

            row_result: dict = {
                "subject": str(row.get("sub_label", "")),
                "correct_object": str(answer_space.iloc[correct_idx]),
                "predicted_object": str(answer_space.iloc[predicted_idx]),
                "p_predicted": float(probs[predicted_idx]),
                "p_predicted_above_chance": round(
                    float(probs[predicted_idx]) * n_cands, 4
                ),
                "p_correct": float(probs[correct_idx]),
                "p_correct_lennorm": float(ln_probs[correct_idx]),
                "p_predicted_lennorm": float(ln_probs[ln_predicted_idx]),
                "is_correct_lennorm": ln_predicted_idx == correct_idx,
                "rank_correct": rank_correct,
                "entropy_norm": round(entropy_norm, 4),
                "is_correct": is_correct,
                "margin": margin,
                "margin_lennorm": margin_lennorm,
                "p_incorrect": p_incorrect,
                "p_incorrect_lennorm": p_incorrect_lennorm,
                # Identity of the strongest non-gold candidate — the "2nd most
                # likely answer" when the model got this template right (its #1
                # pick is then the gold object), or equivalently the model's own
                # top-1 pick when it got the template wrong (predicted_object
                # then coincides with this). Needed downstream (bear cascade
                # attribution) to build the actual contrastive sentence text,
                # not just its score.
                "best_other_object": best_other_object,
            }

            inst_idx = int(row.get("instance_index", row_idx))
            instance_rows.setdefault(inst_idx, []).append(row_result)

        # A subject is learned only when ALL its templates are predicted correctly.
        for inst_idx in sorted(instance_rows):
            tmpl_rows = instance_rows[inst_idx]
            all_correct = all(r["is_correct"] for r in tmpl_rows)

            fact: dict = {
                "subject": tmpl_rows[0]["subject"],
                "correct_object": tmpl_rows[0]["correct_object"],
                "predicted_object": tmpl_rows[0]["predicted_object"],
                "p_predicted": float(np.mean([r["p_predicted"] for r in tmpl_rows])),
                "p_predicted_above_chance": round(
                    float(np.mean([r["p_predicted_above_chance"] for r in tmpl_rows])),
                    4,
                ),
                "p_correct": float(np.mean([r["p_correct"] for r in tmpl_rows])),
                "p_correct_lennorm": float(
                    np.mean([r["p_correct_lennorm"] for r in tmpl_rows])
                ),
                "rank_correct": float(np.mean([r["rank_correct"] for r in tmpl_rows])),
                "entropy_norm": round(
                    float(np.mean([r["entropy_norm"] for r in tmpl_rows])), 4
                ),
                "n_templates_correct": sum(r["is_correct"] for r in tmpl_rows),
            }

            for t_idx, r in enumerate(tmpl_rows):
                bear_items.append(
                    {
                        "relation": rel_code,
                        "subject": r["subject"],
                        "correct_object": r["correct_object"],
                        "template_idx": t_idx,
                        "predicted_object": r["predicted_object"],
                        "p_predicted": r["p_predicted"],
                        "best_other_object": r["best_other_object"],
                        "p_correct": r["p_correct"],
                        "p_correct_lennorm": r["p_correct_lennorm"],
                        "p_predicted_lennorm": r["p_predicted_lennorm"],
                        "is_correct_lennorm": r["is_correct_lennorm"],
                        "entropy_norm": r["entropy_norm"],
                        "rank_correct": r["rank_correct"],
                        "is_correct": r["is_correct"],
                        "margin": r["margin"],
                        "margin_lennorm": r["margin_lennorm"],
                        "p_incorrect": r["p_incorrect"],
                        "p_incorrect_lennorm": r["p_incorrect_lennorm"],
                    }
                )

            (correct_facts if all_correct else wrong_facts).append(fact)

        n = len(correct_facts) + len(wrong_facts)
        acc = len(correct_facts) / n if n else 0.0
        total_correct += len(correct_facts)
        total_n += n

        by_relation[rel_code] = {
            "n": n,
            "accuracy": acc,
            "templates": rel_result.get_metadata("templates"),
            "correct_facts": correct_facts,
            "wrong_facts": wrong_facts,
        }

    return {
        "n": total_n,
        "accuracy": total_correct / total_n if total_n else 0.0,
        "by_relation": by_relation,
        "items": bear_items,
    }


# ── BLiMP download ────────────────────────────────────────────────────────────


def ensure_blimp_files(blimp_dir: Path, phenomena: list[str]) -> None:
    """Download any missing BLiMP JSONL files for the requested phenomena."""
    needed = [
        fname for ph in phenomena if ph in BLIMP_FILES for fname in BLIMP_FILES[ph]
    ]
    missing = [f for f in needed if not (blimp_dir / f).exists()]
    if not missing:
        logger.info(f"All {len(needed)} BLiMP files already present in {blimp_dir}")
        return

    blimp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {len(missing)}/{len(needed)} missing BLiMP files ...")
    for fname in missing:
        url = f"{BLIMP_BASE}/{fname}"
        dest = blimp_dir / fname
        logger.info(f"  {fname}")
        urllib.request.urlretrieve(url, dest)
    logger.info("Download complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]  # bundle root

    parser = argparse.ArgumentParser(
        description="Check whether a pretrained HuggingFace causal LM has learned key features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="HuggingFace model ID or local path (e.g. gpt2, openai-community/gpt2-medium)",
    )
    parser.add_argument(
        "--blimp-dir",
        default=str(repo_root / "data" / "blimp"),
        help="Directory containing BLiMP JSONL files",
    )
    parser.add_argument(
        "--phenomena",
        nargs="+",
        default=ALL_PHENOMENA,
        choices=ALL_PHENOMENA,
        metavar="PHENOMENON",
        help=f"Which phenomena to evaluate. Choices: {ALL_PHENOMENA}",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--output", default=None, help="Write JSON results to this path"
    )
    parser.add_argument(
        "--pairs-output",
        default=None,
        help=(
            "Write per-pair results (phenomenon, file_stem, pair_idx, text, "
            "logprobs, full_sentence_correct, entropy_full_sentence) to this "
            "parquet path. Covers BLiMP phenomena and manual_npi."
        ),
    )
    parser.add_argument(
        "--bear-pairs-output",
        default=None,
        help=(
            "Write per-(fact, template) BEAR results (relation, subject, "
            "correct_object, template_idx, predicted_object, p_predicted, "
            "best_other_object, p_correct, p_correct_lennorm, "
            "p_predicted_lennorm, is_correct_lennorm, entropy_norm, "
            "rank_correct, is_correct, p_incorrect) to this parquet path. "
            "predicted_object/p_predicted are the model's own top-ranked "
            "answer (right or wrong); best_other_object is the strongest "
            "non-gold candidate (at p_incorrect) — the '2nd most likely "
            "answer' when the model got this template right, or equivalently "
            "the model's top-1 pick when it got it wrong. Needed by "
            "prepare/facts.py to recover per-template correctness "
            "counts/scores, since correct_facts/wrong_facts in the JSON "
            "output are aggregated to one row per fact."
        ),
    )
    parser.add_argument(
        "--manual-npi-tsv",
        default=str(repo_root / "data" / "minimal_pairs_npi.tsv"),
        help="Path to the manually constructed NPI minimal-pair set "
        "(data/minimal_pairs_npi.tsv) used for the manual_npi phenomenon",
    )
    parser.add_argument(
        "--token-counts",
        type=int,
        default=None,
        metavar="N",
        help="Print token counts for the first N pairs per file and exit (no model inference)",
    )
    parser.add_argument(
        "--bear-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for BEAR probing (default: 1)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    blimp_dir = Path(args.blimp_dir)

    ensure_blimp_files(blimp_dir, args.phenomena)

    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.token_counts is not None:
        n = args.token_counts
        print(f"\nModel : {args.model}")
        print(f"Token counts — first {n} pairs per file\n")
        for phenomenon in args.phenomena:
            by_file = load_blimp_pairs(blimp_dir, phenomenon)
            for stem, pairs in by_file.items():
                print(f"── {phenomenon} / {stem}")
                header = f"  {'#':>3}  {'good_tok':>8}  {'bad_tok':>8}  {'diff':>5}  good_text"
                print(header)
                print("  " + "-" * (len(header) - 2))
                for i, p in enumerate(pairs[:n]):
                    ids_g = _tokenize(tokenizer, p["text_good"], args.max_length)
                    ids_b = _tokenize(tokenizer, p["text_bad"], args.max_length)
                    ng, nb = len(ids_g), len(ids_b)
                    print(
                        f"  {i:>3}  {ng:>8}  {nb:>8}  {ng - nb:>+5}  {p['text_good']}"
                    )
                print()
        return

    device: str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = cast(
        PreTrainedModel,
        AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32),
    )
    model.to(device)  # type: ignore[arg-type]
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {n_params:,}  |  device: {device}")

    all_results: dict[str, dict] = {}
    all_pair_items: list[dict] = []

    manual_npi_tsv = Path(args.manual_npi_tsv)

    for phenomenon in args.phenomena:
        logger.info(f"\n── {phenomenon} ──")
        if phenomenon == "manual_npi":
            if not manual_npi_tsv.exists():
                logger.warning(f"File not found: {manual_npi_tsv}")
                continue
            pairs = load_manual_npi_pairs(manual_npi_tsv)
            logger.info(f"  {len(pairs)} pairs")
            overall = evaluate_pairs(
                model, tokenizer, pairs, args.max_length, device, collect_per_pair=True
            )
            items = overall.pop("items", [])
            all_results[phenomenon] = {"overall": overall, "items": items}
            if args.pairs_output:
                for it in items:
                    it["phenomenon"] = phenomenon
                    it["file_stem"] = "manual_npi"
                all_pair_items.extend(items)
            continue
        else:
            if phenomenon not in BLIMP_FILES:
                logger.info("  skipped (not in BLIMP_FILES)")
                continue
            by_file = load_blimp_pairs(blimp_dir, phenomenon)
            if not by_file:
                logger.warning(f"No pairs loaded for {phenomenon}")
                continue
            file_results: dict[str, dict] = {}
            for stem, pairs in by_file.items():
                logger.info(f"  {stem}: {len(pairs)} pairs")
                file_results[stem] = evaluate_pairs(
                    model,
                    tokenizer,
                    pairs,
                    args.max_length,
                    device,
                    collect_per_pair=bool(args.pairs_output),
                )
                if args.pairs_output:
                    items = file_results[stem].get("items", [])
                    for it in items:
                        it["phenomenon"] = phenomenon
                        it["file_stem"] = stem
                    all_pair_items.extend(items)
            if len(by_file) > 1:
                all_pairs = [p for ps in by_file.values() for p in ps]
                overall = evaluate_pairs(
                    model, tokenizer, all_pairs, args.max_length, device
                )
            else:
                overall = next(iter(file_results.values()))
            all_results[phenomenon] = {"overall": overall, "by_file": file_results}

    # ── BEAR probing ──────────────────────────────────────────────────────────
    if "facts_bear" in args.phenomena:
        logger.info("\n── bear ──")
        logger.info(
            "  Running BEAR probing via lm-pub-quiz (this may take a while) ..."
        )

        try:
            all_results["bear"] = run_bear_probing(
                model,
                tokenizer,
                device,
                args.model,
                args.bear_batch_size,
            )
            bear_acc = all_results["bear"]["accuracy"]
            bear_n = all_results["bear"]["n"]
            logger.info(f"  BEAR overall accuracy: {bear_acc:.3f} (n={bear_n})")
            bear_items = all_results["bear"].pop("items", [])
            if args.bear_pairs_output:
                if bear_items:
                    Path(args.bear_pairs_output).parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    pd.DataFrame(bear_items).to_parquet(
                        args.bear_pairs_output, index=False
                    )
                    logger.info(f"  Bear per-template pairs → {args.bear_pairs_output}")
                else:
                    logger.warning("--bear-pairs-output given but no items collected")
        except Exception:
            logger.exception("BEAR probing failed; skipping")

    # ── summary table (phenomenon-level) ──────────────────────────────────────
    print(f"\nModel : {args.model}")
    print(f"Params: {n_params:,}  |  device: {device}\n")
    header = f"{'Phenomenon':<25} {'Metric':<15} {'Accuracy':>9} {'N':>6}  {'Source'}"
    print(header)
    print("-" * len(header))

    def _source_tag(m: dict) -> str:
        bn, fn = m.get("blimp_n"), m.get("fallback_n")
        if bn is None:
            return ""
        if fn == 0:
            return "blimp"
        if bn == 0:
            return "fallback"
        return f"blimp:{bn} fallback:{fn}"

    for phenomenon, res in all_results.items():
        if phenomenon == "bear":
            acc = res.get("accuracy", float("nan"))
            n = res.get("n", 0)
            marker = " *" if acc >= 0.50 else "  "
            print(f"{'bear (overall)':<25} {'accuracy':<15} {acc:>9.3f} {n:>6}{marker}")
            print()
            continue
        first = True
        for metric in _METRIC_LABELS:
            m = res["overall"].get(metric)
            if m is None:
                continue
            acc, n = m["acc"], m["n"]
            marker = " *" if acc >= 0.60 else "  "
            phen_col = phenomenon if first else ""
            print(
                f"{phen_col:<25} {metric:<15} {acc:>9.3f} {n:>6}{marker}  {_source_tag(m)}"
            )
            first = False
        print()

    print("* accuracy >= 0.60 — model shows evidence of having learned this feature")

    # ── per-file breakdown ─────────────────────────────────────────────────────
    has_breakdown = any(
        res.get("by_file") for ph, res in all_results.items() if ph != "bear"
    )
    if has_breakdown:
        print(f"\n{'── Per-file breakdown ':-<{len(header)}}")
        for phenomenon, res in all_results.items():
            if phenomenon == "bear" or not res.get("by_file"):
                continue
            for stem, fres in res["by_file"].items():
                print(f"\n{phenomenon} / {stem}")
                for metric in _METRIC_LABELS:
                    m = fres.get(metric)
                    if m is None:
                        continue
                    acc, n = m["acc"], m["n"]
                    marker = " *" if acc >= 0.60 else "  "
                    print(
                        f"  {metric:<15} {acc:>9.3f} {n:>6}{marker}  {_source_tag(m)}"
                    )

    if args.output:
        out = {
            "model": args.model,
            "n_params": n_params,
            "device": device,
            "results": all_results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        logger.info(f"Results → {args.output}")

    if args.pairs_output:
        if not all_pair_items:
            logger.warning("--pairs-output given but no per-pair items were collected")
        else:
            cols = [
                "phenomenon",
                "file_stem",
                "pair_idx",
                "text_good",
                "text_bad",
                "logp_good",
                "logp_bad",
                "full_sentence_correct",
                "softmax_prob_good",
                "learned_x",
                "entropy_full_sentence",
            ]
            df = pd.DataFrame(all_pair_items)
            df = df[cols + [c for c in df.columns if c not in cols]]
            out_path = Path(args.pairs_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            logger.info(f"Per-pair results → {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
