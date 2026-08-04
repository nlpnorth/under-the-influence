"""Filter a corpus based on BEAR facts the model has genuinely learned.

check_pretrained evaluates BEAR per template, so a "fact" here is a
(subject, correct_object) pair within one relation, and a relation has
several templates. A fact qualifies when ALL of the following hold on
EVERY one of the relation's templates (not just one):

  Model conditions (from check_pretrained results JSON):
    1. rank_correct == 1  — model's top prediction is the correct answer
    2. p_correct >= p-correct-threshold  — model assigns enough probability mass
    3. entropy_norm < entropy-threshold  — prediction is confident (not uniform)

  Corpus conditions (from bear_corpus_stats JSON, optional):
    4. subject occurrence count   >= min-subject-count
    5. object occurrence count    >= min-object-count
    6. co-occurrence line count   >= min-cooccur-count

For each qualifying (subject, object) pair, Wikidata aliases are fetched once
and cached.  The filter then marks any sentence where both the subject and
object (or one of their aliases) co-occur — these go into train_matched.

Usage
-----
# Step 1: pre-build the alias cache (run once, NOT as an array job)
python run_bear_facts_filter.py \
    --results /path/to/results.json \
    --alias-cache /path/to/cache.json \
    --prefetch-only

# Step 2: run the corpus filter (slurm array)
python run_bear_facts_filter.py \
    --input  chunk_00.pkl \
    --output /path/to/output/ \
    --results /path/to/results.json \
    --alias-cache /path/to/cache.json \
    --corpus-stats /path/to/bear_corpus_stats.json \
    --corpus-dataset 10b
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_filtering.filters.facts import (
    BearFactsFilter,
    _build_bear_match_fn,
    _build_subject_match_fn,
    get_predefined_aliases,
)
from src.corpus_filtering.pipeline import run_filters

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


# ── Alias overlap helpers ─────────────────────────────────────────────────────


def _aliases_overlap(
    subj_label: str,
    obj_label: str,
    subj_aliases: list[str],
    obj_aliases: list[str],
) -> bool:
    """True if subject and object likely refer to the same or strongly related entity.

    Two checks:
      1. Exact case-insensitive alias intersection — catches Italy/Italian via
         Wikidata P1549 demonyms (Italy's alias list includes "Italian").
      2. Whole-word substring match on the canonical labels — catches
         Mexico/Mexico City (the string "mexico" appears as a whole word in
         "mexico city").
    """
    subj_set = {a.lower() for a in subj_aliases}
    obj_set = {a.lower() for a in obj_aliases}
    if subj_set & obj_set:
        return True
    s, o = subj_label.lower(), obj_label.lower()
    if len(s) >= 4 and re.search(r"\b" + re.escape(s) + r"\b", o):
        return True
    if len(o) >= 4 and re.search(r"\b" + re.escape(o) + r"\b", s):
        return True
    return False


# ── Wikidata helpers ──────────────────────────────────────────────────────────


def _wikidata_search(label: str) -> str | None:
    """Return the top Wikidata Q-ID for an English label, or None."""
    params = urllib.parse.urlencode({
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "limit": 1,
        "format": "json",
    })
    with urllib.request.urlopen(f"{WIKIDATA_API}?{params}", timeout=15) as resp:
        data = json.loads(resp.read())
    results = data.get("search", [])
    return results[0]["id"] if results else None


def _wikidata_aliases(qid: str) -> list[str]:
    """Return English label + all English aliases + demonyms (P1549) for a Q-ID."""
    params = urllib.parse.urlencode({
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels|aliases|claims",
        "languages": "en",
        "format": "json",
    })
    with urllib.request.urlopen(f"{WIKIDATA_API}?{params}", timeout=15) as resp:
        data = json.loads(resp.read())
    entity = data.get("entities", {}).get(qid, {})
    terms: list[str] = []
    label = entity.get("labels", {}).get("en", {}).get("value")
    if label:
        terms.append(label)
    for alias in entity.get("aliases", {}).get("en", []):
        terms.append(alias["value"])
    for claim in entity.get("claims", {}).get("P1549", []):
        mainsnak = claim.get("mainsnak", {})
        if mainsnak.get("snaktype") != "value":
            continue
        val = mainsnak.get("datavalue", {}).get("value", {})
        if isinstance(val, dict) and val.get("language") == "en":
            text = val.get("text")
            if text:
                terms.append(text)
    return terms


def _save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def fetch_aliases(
    labels: list[str],
    cache_path: Path,
    sleep: float = 0.15,
) -> dict[str, list[str]]:
    """Return {label: [surface_form, ...]} for every label, using a JSON cache.

    Labels already in the cache are not re-fetched.  The cache is saved after
    every 50 new lookups and once at the end.
    """
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    missing = [l for l in labels if l not in cache]
    if missing:
        print(f"  Fetching Wikidata aliases for {len(missing)} entities "
              f"({len(labels) - len(missing)} already cached)...")
        for i, label in enumerate(missing, 1):
            try:
                qid = _wikidata_search(label)
                cache[label] = _wikidata_aliases(qid) if qid else [label]
            except Exception as exc:
                print(f"  Warning: failed for '{label}': {exc}")
                cache[label] = [label]
            time.sleep(sleep)
            if i % 50 == 0:
                print(f"  {i}/{len(missing)} fetched — saving cache")
                _save_cache(cache, cache_path)
        _save_cache(cache, cache_path)
        print(f"  Done. Cache saved to {cache_path}")

    return {l: cache.get(l, [l]) for l in labels}


# ── Results parsing ───────────────────────────────────────────────────────────


def load_bear_subject_aliases() -> dict[str, list[str]]:
    """Return {sub_label: [alias, ...]} from the lm_pub_quiz BEAR dataset."""
    from lm_pub_quiz import Dataset
    result: dict[str, list[str]] = {}
    for relation in Dataset.from_name("BEAR"):
        for _, row in relation.instance_table.iterrows():
            label = str(row["sub_label"])
            aliases = row.get("sub_aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            existing = result.setdefault(label, [])
            for a in aliases:
                a = str(a)
                if a not in existing:
                    existing.append(a)
    return result


def load_corpus_stats(
    stats_path: Path | None,
    dataset: str,
) -> dict[tuple[str, str], dict]:
    """Return {(subject, correct_object): {subject, object, cooccur}} for one dataset."""
    if stats_path is None or not stats_path.exists():
        return {}
    with open(stats_path, encoding="utf-8") as f:
        data = json.load(f)
    lookup: dict[tuple[str, str], dict] = {}
    for rel_data in data.get("relations", {}).values():
        for inst in rel_data.get("instances", []):
            key = (inst["subject"], inst["correct_object"])
            lookup[key] = inst.get("by_dataset", {}).get(dataset, {})
    return lookup


def load_qualifying_pairs(
    results_path: Path,
    p_correct_threshold: float,
    entropy_threshold: float,
    corpus_lookup: dict[tuple[str, str], dict],
    min_subject_count: int,
    min_object_count: int,
    min_cooccur_count: int,
) -> list[tuple[str, str, str]]:
    """Return (subject, correct_object, rel_id) triplets that pass all quality gates.

    A pair qualifies only if the model is correct (rank_correct == 1) on
    EVERY template of the relation, and every one of those per-template
    entries individually satisfies the p_correct/entropy_norm thresholds.
    """
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both a full check_pretrained results file and a bare bear dict
    bear = data.get("results", {}).get("bear", data)

    pairs: list[tuple[str, str, str]] = []
    n_total = n_not_all_correct = n_low_p = n_high_entropy = n_corpus = 0
    for rel_id, rel_data in bear.get("by_relation", {}).items():
        # correct_facts already guarantees rank_correct == 1 (per template)
        correct_by_pair: dict[tuple[str, str], list[dict]] = {}
        for fact in rel_data.get("correct_facts", []):
            correct_by_pair.setdefault((fact["subject"], fact["correct_object"]), []).append(fact)
        wrong_pairs = {(fact["subject"], fact["correct_object"]) for fact in rel_data.get("wrong_facts", [])}

        n_total += len(correct_by_pair)
        for key, entries in correct_by_pair.items():
            if key in wrong_pairs:
                n_not_all_correct += 1
                continue
            if any(e.get("p_correct", 0.0) < p_correct_threshold for e in entries):
                n_low_p += 1
                continue
            if any(e.get("entropy_norm", 1.0) >= entropy_threshold for e in entries):
                n_high_entropy += 1
                continue
            if corpus_lookup:
                stats = corpus_lookup.get(key, {})
                if (stats.get("subject", 0) < min_subject_count
                        or stats.get("object", 0) < min_object_count
                        or stats.get("cooccur", 0) < min_cooccur_count):
                    n_corpus += 1
                    continue
            subject, correct_object = key
            pairs.append((subject, correct_object, rel_id))

    print(f"  Candidate facts (correct on >=1 template): {n_total}")
    print(f"  Dropped — not correct on all templates: {n_not_all_correct}, "
          f"low p_correct: {n_low_p}, high entropy: {n_high_entropy}, "
          f"corpus threshold: {n_corpus}")
    print(f"  Qualifying facts: {len(pairs)}")
    return pairs


def _write_excluded_facts(path: Path, pairs: list[tuple[str, str, str]]) -> None:
    """Write the final qualifying (subject, object, rel_id) triplets as a JSON manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"rel_id": rel_id, "subject": subj, "object": obj} for subj, obj, rel_id in pairs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  Excluded facts manifest: {path} ({len(payload)} facts)")


def _fact_slug(rel_id: str, subj: str, obj: str) -> str:
    """Stable filesystem-safe slug for a (rel_id, subject, object) triplet."""
    def _safe(s: str) -> str:
        return re.sub(r"[^a-z0-9_]", "", s.lower().replace(" ", "_").replace("-", "_"))
    return f"{rel_id}__{_safe(subj)}__{_safe(obj)}"


def _write_per_fact_files(
    output_dir: Path,
    pairs: list[tuple[str, str, str]],
    pairs_with_aliases: list[tuple[list[str], list[str]]],
    occurrence_eligible: set[tuple[str, str, str]],
    occurrence_obj_eligible: set[tuple[str, str, str]],
) -> None:
    """Write per-fact matched files under output_dir/BearFacts/facts/<slug>/.

    Each matched line is checked against the fact's cooccurrence match first;
    if it matches, it's filed under <slug>/cooccurrence/train_matched.txt.
    Otherwise, if the fact is subject-occurrence-eligible and the line merely
    mentions the subject, it's filed under <slug>/subj_occurrence/train_matched.txt.
    Otherwise, if the fact is object-occurrence-eligible and the line merely
    mentions the object, it's filed under <slug>/obj_occurrence/train_matched.txt.
    The three files are disjoint per fact by construction.
    """
    matched_txt = output_dir / "BearFacts" / "train_matched.txt"
    if not matched_txt.exists():
        print(f"  Per-fact split: no train_matched.txt found at {matched_txt}, skipping.")
        return

    lines = [l for l in matched_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print("  Per-fact split: train_matched.txt is empty, skipping.")
        return

    # Build per-fact match functions (cooccurrence always; subject/object-only when eligible)
    entries = []
    for (subj, obj, rel_id), (subj_terms, obj_terms) in zip(pairs, pairs_with_aliases):
        slug = _fact_slug(rel_id, subj, obj)
        cooccur_fn = _build_bear_match_fn([(subj_terms, obj_terms)])
        subj_fn = (
            _build_subject_match_fn([subj_terms])
            if (subj, obj, rel_id) in occurrence_eligible
            else None
        )
        obj_fn = (
            _build_subject_match_fn([obj_terms])
            if (subj, obj, rel_id) in occurrence_obj_eligible
            else None
        )
        entries.append((slug, cooccur_fn, subj_fn, obj_fn))

    cooccur_lines: dict[str, list[str]] = {}
    subj_occur_lines: dict[str, list[str]] = {}
    obj_occur_lines: dict[str, list[str]] = {}
    for line in lines:
        for slug, cooccur_fn, subj_fn, obj_fn in entries:
            if cooccur_fn(line):
                cooccur_lines.setdefault(slug, []).append(line)
            elif subj_fn is not None and subj_fn(line):
                subj_occur_lines.setdefault(slug, []).append(line)
            elif obj_fn is not None and obj_fn(line):
                obj_occur_lines.setdefault(slug, []).append(line)

    def _write(bucket: dict[str, list[str]], subdir: str) -> int:
        n = 0
        for slug, matched in bucket.items():
            out = output_dir / "BearFacts" / "facts" / slug / subdir
            out.mkdir(parents=True, exist_ok=True)
            (out / "train_matched.txt").write_text("\n".join(matched) + "\n", encoding="utf-8")
            n += 1
        return n

    n_cooccur = _write(cooccur_lines, "cooccurrence")
    n_subj_occur = _write(subj_occur_lines, "subj_occurrence")
    n_obj_occur = _write(obj_occur_lines, "obj_occurrence")
    print(f"  Per-fact files: {n_cooccur}/{len(entries)} facts have cooccurrence matches, "
          f"{n_subj_occur}/{len(entries)} facts have subj_occurrence matches, "
          f"{n_obj_occur}/{len(entries)} facts have obj_occurrence matches")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a corpus based on BEAR facts the model genuinely learned.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", help="Corpus pickle or CoNLL-U file/dir")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--results", required=True,
        help="Path to check_pretrained results JSON (must contain bear.by_relation)",
    )
    parser.add_argument(
        "--alias-cache", default=None, metavar="PATH",
        help="Path to Wikidata alias cache JSON. "
             "Defaults to wikidata_alias_cache.json next to the results file.",
    )
    parser.add_argument(
        "--p-correct-threshold", type=float, default=0.1, metavar="T",
        help="Minimum p_correct for a fact to qualify (model's probability on the correct answer).",
    )
    parser.add_argument(
        "--entropy-threshold", type=float, default=0.8, metavar="T",
        help="Maximum entropy_norm for a fact to qualify (lower = more confident).",
    )
    parser.add_argument(
        "--corpus-stats", default=None, metavar="PATH",
        help="Path to bear_corpus_stats.json produced by bear_corpus_cooccurrence.py. "
             "When provided, corpus occurrence thresholds are applied.",
    )
    parser.add_argument(
        "--corpus-dataset", default="10b", metavar="DATASET",
        help="Which dataset column to use from --corpus-stats (default: 10b).",
    )
    parser.add_argument(
        "--min-subject-count", type=int, default=5, metavar="N",
        help="Minimum corpus lines containing the subject.",
    )
    parser.add_argument(
        "--min-object-count", type=int, default=5, metavar="N",
        help="Minimum corpus lines containing the correct object.",
    )
    parser.add_argument(
        "--min-cooccur-count", type=int, default=1, metavar="N",
        help="Minimum corpus lines where subject and object co-occur.",
    )
    parser.add_argument(
        "--excluded-facts-output", default=None, metavar="PATH",
        help="If set, write the final qualifying (excluded-from-training) facts as a JSON "
             "manifest to this path.",
    )
    parser.add_argument(
        "--prefetch-only", action="store_true",
        help="Only build the alias cache and exit — do not run the corpus filter. "
             "Run this once before submitting the slurm array.",
    )
    parser.add_argument(
        "--exclude-alias-overlap", action="store_true",
        help="Remove facts whose subject and object share a Wikidata alias or where "
             "one canonical label contains the other as a whole word (e.g. Italy/Italian "
             "via P1549 demonym, Mexico/Mexico City via substring). Applied after alias "
             "cache is built.",
    )
    parser.add_argument(
        "--occurrence-subject-count-threshold", type=int, default=None, metavar="N",
        help="If set, also remove every sentence merely mentioning the subject (not just "
             "cooccurring lines) for qualifying facts whose corpus subject-count is below "
             "this threshold. Requires --corpus-stats. Written to a separate "
             "<slug>/subj_occurrence/ subfolder per fact, disjoint from <slug>/cooccurrence/.",
    )
    parser.add_argument(
        "--occurrence-object-count-threshold", type=int, default=None, metavar="N",
        help="If set, also remove every sentence merely mentioning the object (not just "
             "cooccurring lines) for qualifying facts whose corpus object-count is below "
             "this threshold. Requires --corpus-stats. Written to a separate "
             "<slug>/obj_occurrence/ subfolder per fact, disjoint from <slug>/cooccurrence/ "
             "and <slug>/subj_occurrence/.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    cache_path = (
        Path(args.alias_cache)
        if args.alias_cache
        else results_path.parent / "wikidata_alias_cache.json"
    )

    # ── 1. Extract qualifying facts ───────────────────────────────────────────
    print(f"Loading results from {results_path}")
    corpus_stats_path = Path(args.corpus_stats) if args.corpus_stats else None
    corpus_lookup = load_corpus_stats(corpus_stats_path, args.corpus_dataset)
    if corpus_stats_path:
        print(f"Corpus stats: {corpus_stats_path} (dataset={args.corpus_dataset}, "
              f"{len(corpus_lookup)} entries)")
    pairs = load_qualifying_pairs(
        results_path,
        p_correct_threshold=args.p_correct_threshold,
        entropy_threshold=args.entropy_threshold,
        corpus_lookup=corpus_lookup,
        min_subject_count=args.min_subject_count,
        min_object_count=args.min_object_count,
        min_cooccur_count=args.min_cooccur_count,
    )
    if not pairs:
        print("No qualifying facts after filtering — nothing to do.")
        return

    # ── 2. Fetch aliases ──────────────────────────────────────────────────────
    all_labels = list({label for subj, obj, _ in pairs for label in (subj, obj)})
    print(f"Unique entity labels: {len(all_labels)}")
    print("Loading BEAR subject aliases from lm_pub_quiz ...")
    bear_aliases = load_bear_subject_aliases()
    print(f"  {len(bear_aliases)} subjects with aliases")
    alias_map = fetch_aliases(all_labels, cache_path)

    # ── 2b. Alias overlap filter ──────────────────────────────────────────────
    if args.exclude_alias_overlap:
        n_before = len(pairs)
        pairs = [
            (subj, obj, rel_id) for subj, obj, rel_id in pairs
            if not _aliases_overlap(
                subj, obj,
                alias_map.get(subj, [subj]),
                alias_map.get(obj, [obj]),
            )
        ]
        print(f"  Alias-overlap filter: {n_before - len(pairs)} pairs removed, "
              f"{len(pairs)} remain")
        if not pairs:
            print("No qualifying facts after alias-overlap check — exiting.")
            return

    if args.excluded_facts_output:
        _write_excluded_facts(Path(args.excluded_facts_output), pairs)

    if args.prefetch_only:
        print("--prefetch-only: alias cache built. Exiting.")
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required when not using --prefetch-only")

    # ── 3. Build filter and run pipeline ─────────────────────────────────────
    def _merge(label: str) -> list[str]:
        wikidata = alias_map.get(label, [label])
        bear = bear_aliases.get(label, [])
        predefined = get_predefined_aliases(label)
        return list(dict.fromkeys(wikidata + bear + predefined))  # deduplicate, wikidata first

    pairs_with_aliases = [
        (_merge(subj), _merge(obj))
        for subj, obj, _ in pairs
    ]

    occurrence_eligible: set[tuple[str, str, str]] = set()
    occurrence_subj_terms: list[list[str]] = []
    if args.occurrence_subject_count_threshold is not None:
        threshold = args.occurrence_subject_count_threshold
        for (subj, obj, rel_id), (subj_terms, _obj_terms) in zip(pairs, pairs_with_aliases):
            stats = corpus_lookup.get((subj, obj), {})
            if stats.get("subject", 0) < threshold:
                occurrence_eligible.add((subj, obj, rel_id))
                occurrence_subj_terms.append(subj_terms)
        print(f"  Occurrence-eligible facts (subject_count < {threshold}): "
              f"{len(occurrence_eligible)} / {len(pairs)}")

    occurrence_obj_eligible: set[tuple[str, str, str]] = set()
    occurrence_obj_terms: list[list[str]] = []
    if args.occurrence_object_count_threshold is not None:
        threshold = args.occurrence_object_count_threshold
        for (subj, obj, rel_id), (_subj_terms, obj_terms) in zip(pairs, pairs_with_aliases):
            stats = corpus_lookup.get((subj, obj), {})
            if stats.get("object", 0) < threshold:
                occurrence_obj_eligible.add((subj, obj, rel_id))
                occurrence_obj_terms.append(obj_terms)
        print(f"  Occurrence-eligible facts (object_count < {threshold}): "
              f"{len(occurrence_obj_eligible)} / {len(pairs)}")

    bear_filter = BearFactsFilter(
        pairs_with_aliases,
        occurrence_subj_terms or None,
        occurrence_obj_terms or None,
    )
    print(f"\nRunning BearFacts filter over {args.input}")
    run_filters([bear_filter], args.input, output_dir=args.output)

    # Write per-fact matched files under
    # BearFacts/facts/<slug>/{cooccurrence,subj_occurrence,obj_occurrence}/
    print(f"\nWriting per-fact matched files ...")
    _write_per_fact_files(
        Path(args.output), pairs, pairs_with_aliases,
        occurrence_eligible, occurrence_obj_eligible,
    )


if __name__ == "__main__":
    main()
