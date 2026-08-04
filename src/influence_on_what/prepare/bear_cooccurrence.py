#!/usr/bin/env python3
"""
Count corpus occurrences for every BEAR benchmark instance:
  - lines containing subject (or any alias)
  - lines containing correct_object (or any alias)
  - lines containing both (co-occurrence on same line)

Matching mirrors the logic in corpus_filtering/src/corpus_filtering/filters/facts.py:
short all-caps tokens (US, UK, …) are matched case-sensitively; everything else
case-insensitively, with Unicode normalization as a fallback.

Broken down by training corpus size (50m / 100m / 1b / 10b).

Output: data/bear_corpus_stats.json

Run from anywhere in the project:
  python python -m influence_on_what.prepare.bear_cooccurrence
  python python -m influence_on_what.prepare.bear_cooccurrence --data-root /alt/path --alias-cache /path/to/cache.json
  python python -m influence_on_what.prepare.bear_cooccurrence --models 50m 100m
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Wikimedia's User-Agent policy asks scripts calling their APIs to identify
# themselves with a tool name and a contact address, so they can get in touch
# if a script misbehaves; requests with a generic or absent User-Agent may be
# rate-limited or refused.  Set WIKIDATA_CONTACT to your own address before
# running this against the live API.
#
# Note that this only matters when regenerating the statistics for a corpus
# whose entities are not already in the alias cache — ensure_alias_cache()
# requests only the labels it is missing, and the shipped
# data/wikidata_alias_cache.json covers the full BEAR benchmark.
WIKIDATA_CONTACT = os.environ.get("WIKIDATA_CONTACT", "").strip()
WIKIDATA_UA = (
    f"InfluenceOnWhat/1.0 ({WIKIDATA_CONTACT})"
    if WIKIDATA_CONTACT
    else "InfluenceOnWhat/1.0 (research use; set WIKIDATA_CONTACT to your contact address)"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[3] / "data" / "bear_corpus_stats.json"
DEFAULT_DATA_ROOT = Path("/home/argy/english_data/filter_output")
CORPUS_SUBPATH = "Binding-reflexive/train_full.txt"

# Chunks used per model size (matching goldfish_attribute.slurm)
CHUNKS_BY_MODEL: dict[str, list[str] | None] = {
    "50m": ["00"],
    "100m": ["00", "01"],
    "1b": [
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "12",
        "13",
        "14",
        "15",
        "16",
    ],
    "10b": None,  # auto-detect from DATA_ROOT
    "wikipedia": None,  # auto-detect from DATA_ROOT
}

# ── Wikidata helpers ──────────────────────────────────────────────────────────


def _wikidata_get(params: dict) -> dict:
    url = f"{WIKIDATA_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": WIKIDATA_UA})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _wikidata_search(label: str) -> str | None:
    data = _wikidata_get(
        {
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "limit": 1,
            "format": "json",
        }
    )
    results = data.get("search", [])
    return results[0]["id"] if results else None


def _wikidata_aliases(qid: str) -> list[str]:
    data = _wikidata_get(
        {
            "action": "wbgetentities",
            "ids": qid,
            "props": "labels|aliases|claims",
            "languages": "en",
            "format": "json",
        }
    )
    entity = data.get("entities", {}).get(qid, {})
    terms: list[str] = []
    label_val = entity.get("labels", {}).get("en", {}).get("value")
    if label_val:
        terms.append(label_val)
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


def ensure_alias_cache(
    labels: list[str],
    cache: dict[str, list[str]],
    cache_path: Path,
    sleep: float = 0.15,
) -> dict[str, list[str]]:
    """Fetch Wikidata aliases for any labels missing from cache; save and return updated cache."""
    missing = [l for l in labels if l not in cache]
    if not missing:
        return cache
    print(
        f"  Fetching Wikidata aliases for {len(missing)} entities "
        f"({len(labels) - len(missing)} already cached) ...",
        file=sys.stderr,
    )
    for i, label in enumerate(missing, 1):
        try:
            qid = _wikidata_search(label)
            cache[label] = _wikidata_aliases(qid) if qid else [label]
        except Exception as exc:
            print(f"  Warning: failed for '{label}': {exc}", file=sys.stderr)
            cache[label] = [label]
        time.sleep(sleep)
        if i % 50 == 0:
            print(f"  {i}/{len(missing)} fetched — saving cache", file=sys.stderr)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    print(f"  Alias cache saved to {cache_path}", file=sys.stderr)
    return cache


# ── corpus_filtering helpers ──────────────────────────────────────────────────


def _import_facts_helpers():
    corpus_filtering_src = REPO_ROOT / "vendor" / "corpus_filtering" / "src"
    if str(corpus_filtering_src) not in sys.path:
        sys.path.insert(0, str(corpus_filtering_src))
    from corpus_filtering.filters.facts import (  # type: ignore
        _split_case,
        get_predefined_aliases,
    )

    return _split_case, get_predefined_aliases


# ── article stripping ────────────────────────────────────────────────────────

_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _bare_form(s: str) -> str | None:
    """Strip leading article; return bare form only if remainder is all-caps or multi-word."""
    m = _ARTICLE_RE.match(s)
    if not m:
        return None
    bare = s[m.end() :]
    if bare.isupper() or " " in bare:
        return bare
    return None


# ── data loading ──────────────────────────────────────────────────────────────


def get_chunks(data_root: Path, model: str) -> list[str]:
    if CHUNKS_BY_MODEL[model] is not None:
        return CHUNKS_BY_MODEL[model]
    return sorted(
        p.name[len("chunk_") :]
        for p in data_root.iterdir()
        if p.name.startswith("chunk_") and p.is_dir()
    )


def load_bear_instances() -> list[dict]:
    from lm_pub_quiz import Dataset

    dataset = Dataset.from_name("BEAR")
    instances: list[dict] = []
    for relation in dataset:
        rel = relation.relation_code
        answer_space = relation.answer_space
        for _, row in relation.instance_table.iterrows():
            aliases = row.get("sub_aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            instances.append(
                {
                    "rel": rel,
                    "subject": str(row["sub_label"]),
                    "correct_object": str(answer_space.iloc[int(row["answer_idx"])]),
                    "sub_aliases": [str(a) for a in aliases],
                }
            )
    return instances


def load_alias_cache(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Common English function words (articles, prepositions, pronouns, modal/
# auxiliary verbs, conjunctions) that some entities' alias lists include as a
# literal short form — e.g. India's BEAR sub_aliases contains a lowercase
# "in" alongside "IN". _is_case_sensitive()'s short-all-caps guard only looks
# at each alias's own casing, so the lowercase "in" alias slips through as
# case-insensitive and matches the preposition "in" in a huge fraction of any
# English corpus (verified: ~42% of a sampled chunk). No amount of case
# sensitivity fixes this — the token itself just isn't a distinguishing
# surface form — so these are dropped outright before indexing. Only applied
# to aliases, never to the canonical label itself.
_FUNCTION_WORD_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "to", "of", "or", "and", "but",
    "if", "is", "it", "be", "as", "we", "us", "i", "you", "he", "she", "they",
    "this", "that", "can", "may", "will", "shall", "do", "does", "did", "has",
    "have", "had", "not", "no", "so", "up", "out", "off", "for", "from",
    "with", "into", "onto", "than", "then", "when", "was", "were", "are",
    "am", "my", "me", "him", "her", "his", "its", "our", "your", "their",
}


def _is_function_word(term: str) -> bool:
    return term.strip().lower() in _FUNCTION_WORD_STOPWORDS


def collect_surface_forms(
    label: str,
    extra_aliases: list[str],
    alias_cache: dict[str, list[str]],
    get_predefined_aliases,
) -> list[str]:
    """Canonical label + BEAR sub_aliases + Wikidata cache + predefined + bare article forms."""
    forms: list[str] = [label]
    aliases = [
        f
        for f in (
            list(extra_aliases)
            + alias_cache.get(label, [])
            + get_predefined_aliases(label)
        )
        if not _is_function_word(f)
    ]
    forms.extend(aliases)
    forms.extend(
        x
        for f in list(forms)
        if (x := _bare_form(f)) is not None and not _is_function_word(x)
    )
    return list(dict.fromkeys(f for f in forms if f))  # deduplicate, preserve order


# ── index building ────────────────────────────────────────────────────────────


def build_indices(instances: list[dict], alias_cache: dict) -> dict:
    """
    Returns a dict with:
      sub_cs_idx / sub_ci_idx / obj_cs_idx / obj_ci_idx
        — term → [instance_idx, ...]
      max_words
        — longest term length in tokens (caps n-gram scan per line)
      first_words
        — set of first tokens from all ci terms (pre-filter)
    """
    _split_case, get_predefined_aliases = _import_facts_helpers()

    sub_cs_idx: dict[str, list[int]] = defaultdict(list)
    sub_ci_idx: dict[str, list[int]] = defaultdict(list)
    obj_cs_idx: dict[str, list[int]] = defaultdict(list)
    obj_ci_idx: dict[str, list[int]] = defaultdict(list)

    for i, inst in enumerate(instances):
        sub_forms = collect_surface_forms(
            inst["subject"],
            inst.get("sub_aliases", []),
            alias_cache,
            get_predefined_aliases,
        )
        obj_forms = collect_surface_forms(
            inst["correct_object"], [], alias_cache, get_predefined_aliases
        )
        s_cs, s_ci = _split_case(sub_forms)
        o_cs, o_ci = _split_case(obj_forms)
        for t in s_cs:
            sub_cs_idx[t].append(i)
        for t in s_ci:
            sub_ci_idx[t].append(i)
        for t in o_cs:
            obj_cs_idx[t].append(i)
        for t in o_ci:
            obj_ci_idx[t].append(i)

    all_terms = set(sub_cs_idx) | set(sub_ci_idx) | set(obj_cs_idx) | set(obj_ci_idx)
    max_words = max((len(t.split()) for t in all_terms), default=1)

    return {
        "sub_cs_idx": dict(sub_cs_idx),
        "sub_ci_idx": dict(sub_ci_idx),
        "obj_cs_idx": dict(obj_cs_idx),
        "obj_ci_idx": dict(obj_ci_idx),
        "max_words": max_words,
        "first_words": {t.split()[0] for t in sub_ci_idx}
        | {t.split()[0] for t in obj_ci_idx},
    }


# ── corpus scanning ───────────────────────────────────────────────────────────


def scan_chunk(
    instances: list[dict],
    idx: dict,
    corpus_file: Path,
    label: str = "",
    pos: str = "",
) -> list[dict]:
    """Scan a single corpus file; return per-instance counts."""
    counts: list[dict] = [{"subject": 0, "object": 0, "cooccur": 0} for _ in instances]
    if not corpus_file.exists():
        print(f"  WARNING: {corpus_file} not found, skipping", file=sys.stderr)
        return counts

    sub_cs_idx = idx["sub_cs_idx"]
    sub_ci_idx = idx["sub_ci_idx"]
    obj_cs_idx = idx["obj_cs_idx"]
    obj_ci_idx = idx["obj_ci_idx"]
    max_words = idx["max_words"]
    first_words = idx["first_words"]

    LOG_EVERY = 200_000
    tag = label or corpus_file.parent.parent.name
    pos_str = f"[{pos}] " if pos else ""
    total_lines = matched_lines = 0
    start = time.time()

    print(
        f"  {pos_str}scanning {tag} ... ({time.strftime('%H:%M:%S')})",
        file=sys.stderr,
        flush=True,
    )
    with open(corpus_file, errors="replace") as f:
        for line in f:
            total_lines += 1

            if total_lines % LOG_EVERY == 0:
                elapsed = time.time() - start
                lps = total_lines / elapsed if elapsed > 0 else 0
                print(
                    f"    {tag}: {total_lines / 1e6:.1f}M lines | "
                    f"{lps / 1e3:.1f}k lines/sec | elapsed {elapsed:.0f}s",
                    file=sys.stderr,
                    flush=True,
                )

            words = re.findall(r"\w+", line)
            words_lower = [w.lower() for w in words]
            if not (set(words_lower) & first_words):
                continue

            matched_sub: set[int] = set()
            matched_obj: set[int] = set()

            n = len(words)
            for s in range(n):
                for length in range(1, min(max_words, n - s) + 1):
                    ci = " ".join(words_lower[s : s + length])
                    cs = " ".join(words[s : s + length])
                    for i in sub_ci_idx.get(ci, ()):
                        matched_sub.add(i)
                    for i in sub_cs_idx.get(cs, ()):
                        matched_sub.add(i)
                    for i in obj_ci_idx.get(ci, ()):
                        matched_obj.add(i)
                    for i in obj_cs_idx.get(cs, ()):
                        matched_obj.add(i)

            if not matched_sub and not matched_obj:
                continue
            matched_lines += 1
            for i in matched_sub:
                counts[i]["subject"] += 1
                if i in matched_obj:
                    counts[i]["cooccur"] += 1
            for i in matched_obj:
                counts[i]["object"] += 1

    elapsed = time.time() - start
    lps = total_lines / elapsed if elapsed > 0 else 0
    print(
        f"    {tag} done: {total_lines / 1e6:.2f}M lines in {elapsed:.0f}s "
        f"({lps / 1e3:.1f}k lines/sec, {matched_lines:,} matched)",
        file=sys.stderr,
        flush=True,
    )
    return counts


def count_corpus(
    instances: list[dict],
    idx: dict,
    data_root: Path,
    model: str,
    override_chunks: list[str] | None = None,
    corpus_subpath: str = CORPUS_SUBPATH,
) -> list[dict]:
    """Sequential scan of all chunks for a model (convenience wrapper around scan_chunk)."""
    chunks = (
        override_chunks if override_chunks is not None else get_chunks(data_root, model)
    )
    merged = [{"subject": 0, "object": 0, "cooccur": 0} for _ in instances]
    scan_start = time.time()
    for chunk_idx, chunk in enumerate(chunks, 1):
        corpus_file = data_root / f"chunk_{chunk}" / corpus_subpath
        c = scan_chunk(
            instances,
            idx,
            corpus_file,
            label=f"chunk_{chunk}",
            pos=f"{chunk_idx}/{len(chunks)}",
        )
        for i in range(len(instances)):
            merged[i]["subject"] += c[i]["subject"]
            merged[i]["object"] += c[i]["object"]
            merged[i]["cooccur"] += c[i]["cooccur"]
    print(
        f"  {model}: all chunks done in {time.time() - scan_start:.0f}s",
        file=sys.stderr,
    )
    return merged


def merge_all(
    partial_dir: Path,
    instances: list[dict],
    models: list[str],
    output_path: Path,
) -> None:
    """Assemble final output from per-chunk partial files for each model."""
    model_counts: dict[str, list[dict]] = {}
    for model in models:
        chunk_ids = CHUNKS_BY_MODEL[model] or sorted(
            pf.stem for pf in partial_dir.glob("*.json")
        )
        merged = [{"subject": 0, "object": 0, "cooccur": 0} for _ in instances]
        missing = []
        for chunk_id in chunk_ids:
            pf = partial_dir / f"{chunk_id}.json"
            if not pf.exists():
                missing.append(chunk_id)
                continue
            for i, c in enumerate(json.loads(pf.read_text())["counts"]):
                merged[i]["subject"] += c["subject"]
                merged[i]["object"] += c["object"]
                merged[i]["cooccur"] += c["cooccur"]
        if missing:
            print(f"  WARNING: {model} missing chunks: {missing}", file=sys.stderr)
        print(
            f"  {model}: merged {len(chunk_ids) - len(missing)}/{len(chunk_ids)} chunks",
            file=sys.stderr,
        )
        model_counts[model] = merged

    by_relation: dict[str, list] = defaultdict(list)
    for i, inst in enumerate(instances):
        by_relation[inst["rel"]].append(
            {
                "subject": inst["subject"],
                "correct_object": inst["correct_object"],
                "by_dataset": {model: model_counts[model][i] for model in models},
            }
        )

    output = {
        "relations": {rel: {"instances": v} for rel, v in sorted(by_relation.items())}
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {output_path}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        type=Path,
        help="Root dir containing chunk_XX subdirectories",
    )
    parser.add_argument(
        "--corpus-subpath",
        default=CORPUS_SUBPATH,
        metavar="PATH",
        help="Path within each chunk_XX dir to the plain-text corpus file "
        f"(default: {CORPUS_SUBPATH!r}, matching argy's filter_output layout; "
        "pass e.g. 'train_full.txt' for a flat chunk_XX/train_full.txt layout)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(CHUNKS_BY_MODEL),
        choices=list(CHUNKS_BY_MODEL),
        metavar="MODEL",
        help="Which model corpus sizes to process (default: all)",
    )
    parser.add_argument(
        "--alias-cache",
        default=None,
        type=Path,
        metavar="PATH",
        help="Wikidata alias cache JSON. Built automatically if missing when this flag is set.",
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help="Only build the Wikidata alias cache and exit (no corpus needed). "
        "Run this locally, then scp the cache to the cluster before submitting.",
    )
    parser.add_argument(
        "--expand-cache",
        action="store_true",
        help="Append missing bare-form variants (strip leading 'the/a/an' when remainder "
        "is all-caps or multi-word) to --alias-cache and exit.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), type=Path)
    parser.add_argument(
        "--chunk-id",
        metavar="CHUNK",
        help="Scan exactly this chunk and write a partial file (array job mode)",
    )
    parser.add_argument(
        "--partial-dir",
        type=Path,
        metavar="DIR",
        help="Directory for per-chunk partial JSON files (used with --chunk-id or --merge-partials)",
    )
    parser.add_argument(
        "--merge-partials",
        action="store_true",
        help="Merge partials from --partial-dir into --output for all --models and exit",
    )
    args = parser.parse_args()

    print("Loading BEAR instances ...", file=sys.stderr)
    instances = load_bear_instances()
    n_rels = len({i["rel"] for i in instances})
    print(f"  {len(instances):,} instances across {n_rels} relations", file=sys.stderr)

    if args.merge_partials:
        if args.partial_dir is None:
            print("ERROR: --merge-partials requires --partial-dir", file=sys.stderr)
            sys.exit(1)
        merge_all(args.partial_dir, instances, args.models, args.output)
        return

    print("Loading alias cache ...", file=sys.stderr)
    alias_cache = load_alias_cache(args.alias_cache)
    print(f"  {len(alias_cache)} entities in cache", file=sys.stderr)

    if args.alias_cache is not None:
        all_labels = list(
            {inst["subject"] for inst in instances}
            | {inst["correct_object"] for inst in instances}
        )
        alias_cache = ensure_alias_cache(all_labels, alias_cache, args.alias_cache)

    if args.prefetch_only:
        print("--prefetch-only: alias cache built. Exiting.", file=sys.stderr)
        return

    if args.expand_cache:
        if args.alias_cache is None:
            print("ERROR: --expand-cache requires --alias-cache PATH", file=sys.stderr)
            sys.exit(1)
        added = 0
        for aliases in alias_cache.values():
            extras = [
                b for a in list(aliases) if (b := _bare_form(a)) and b not in aliases
            ]
            aliases.extend(extras)
            added += len(extras)
        args.alias_cache.write_text(
            json.dumps(alias_cache, indent=2, ensure_ascii=False)
        )
        print(
            f"--expand-cache: {added} bare forms added, cache saved to {args.alias_cache}",
            file=sys.stderr,
        )
        return

    print("Building term index ...", file=sys.stderr)
    idx = build_indices(instances, alias_cache)
    n_sub = len(idx["sub_cs_idx"]) + len(idx["sub_ci_idx"])
    n_obj = len(idx["obj_cs_idx"]) + len(idx["obj_ci_idx"])
    print(
        f"  {n_sub} subject surface forms, {n_obj} object surface forms",
        file=sys.stderr,
    )

    if args.chunk_id is not None:
        # Array task mode: scan one chunk, write partial, exit
        if args.partial_dir is None:
            print("ERROR: --chunk-id requires --partial-dir", file=sys.stderr)
            sys.exit(1)
        corpus_file = args.data_root / f"chunk_{args.chunk_id}" / args.corpus_subpath
        counts = scan_chunk(instances, idx, corpus_file, label=f"chunk_{args.chunk_id}")
        args.partial_dir.mkdir(parents=True, exist_ok=True)
        out_file = args.partial_dir / f"{args.chunk_id}.json"
        out_file.write_text(json.dumps({"chunk": args.chunk_id, "counts": counts}))
        print(f"Partial written to {out_file}", file=sys.stderr)
        return

    # Sequential mode: scan all chunks for each model and write final output directly
    model_counts: dict[str, list[dict]] = {}
    for model in args.models:
        print(f"\n=== {model} ===", file=sys.stderr)
        model_counts[model] = count_corpus(
            instances, idx, args.data_root, model, corpus_subpath=args.corpus_subpath
        )

    by_relation: dict[str, list] = defaultdict(list)
    for i, inst in enumerate(instances):
        by_relation[inst["rel"]].append(
            {
                "subject": inst["subject"],
                "correct_object": inst["correct_object"],
                "by_dataset": {model: model_counts[model][i] for model in args.models},
            }
        )

    output = {
        "relations": {rel: {"instances": v} for rel, v in sorted(by_relation.items())}
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
