# data/

Everything the pipeline reads that is not a corpus. The two probing benchmarks
are included verbatim, so no step needs network access.

## Included benchmarks

### `blimp/` — BLiMP paradigm files

11 `.jsonl` files, 1,000 minimal pairs each — exactly the paradigms named in
`BLIMP_FILES_FILTERED` in `../src/influence_on_what/probe_models.py`, which is
what `BLIMP_FILES` points at: one file each for `binding` and `island_effects`,
two for `quantifiers`, and the seven `npi_licensing` files (kept for
completeness; the NPI phenomenon is evaluated on `minimal_pairs_npi.tsv`
instead, for the reasons in the top-level README). The other 56 BLiMP paradigms
are unused here — flipping `BLIMP_FILES` to `BLIMP_FILES_ALL` downloads them.

- Source: <https://github.com/alexwarstadt/blimp> (`data/`), mirrored as
  <https://huggingface.co/datasets/nyu-mll/blimp>
- Paper: Warstadt et al. (2020), *BLiMP: The Benchmark of Linguistic Minimal
  Pairs for English*, TACL 8.
- License: CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>)

`probe_models.py` and `prepare/linguistic.py` download any paradigm file that is
missing from this directory, so adding a phenomenon needs no manual fetch.

### `lm-pub-quiz/datasets/bear/` — BEAR knowledge probe

The BEAR (not BEAR-big) split: 60 relation files, 7,731 entity–relation
instances, three templates each, plus `metadata_relations.json`.

- Source: <https://github.com/lm-pub-quiz/BEAR>, subdirectory `BEAR/`, pinned
  commit `725b4e3139d0a5fdf914b0419ba744273dddc689` — the revision the
  `lm-pub-quiz` package downloads for `Dataset.from_name("BEAR")`.
- Paper: Wiland et al. (2024), *BEAR: A Unified Framework for Evaluating
  Relational Knowledge in Causal and Masked Language Models*, NAACL Findings.
- License: CC BY-SA 4.0, full text in `lm-pub-quiz/LICENSE-BEAR.txt`

The directory layout mirrors the `lm-pub-quiz` cache, so
`Dataset.from_name("BEAR")` resolves to this copy once
`LM_PUB_QUIZ_CACHE_ROOT` points at `data/lm-pub-quiz/`. Every script in `run/`
sets that variable via `config/env.sh`. When invoking a module directly, either
export it yourself:

```bash
export LM_PUB_QUIZ_CACHE_ROOT="$PWD/data/lm-pub-quiz"
```

or let `lm-pub-quiz` download the same pinned revision into `~/.lm-pub-quiz/`.

## Derived and hand-built files

- `minimal_pairs_npi.tsv` — the NPI minimal-pair set built for this work (used
  instead of BLiMP's NPI suite; the reason is in the top-level README §Data).
- `bear_corpus_stats.json`, `bear_corpus_stats_wikipedia.json` — per-fact entity
  occurrence and co-occurrence counts behind the "corpus-supported" (≥10
  co-occurrences) criterion. Regenerate with
  `python -m influence_on_what.prepare.bear_cooccurrence`.
- `wikidata_alias_cache.json` — cached Wikidata labels, aliases and demonyms, so
  the fact filter does not re-query the API.

## Not included

The corpora. Common Corpus and the English Wikipedia dump are not redistributed;
see the top-level README §Data, and `prepare/setup_wikipedia_data.py` for the
Wikipedia download and segmentation step.
