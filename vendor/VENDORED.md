# Vendored third-party code

Three dependencies are copied into this bundle rather than pinned as remote
dependencies, so that it is self-contained and reviewable without network
access. Each is listed below with its provenance and the modifications made.

## `bergson/`

The influence-function library used for all gradient collection,
preconditioning and scoring.

- Upstream: <https://github.com/EleutherAI/bergson>
- Vendored from commit `2bb11a7414547fdb6f62f88ab6b41730efbd9ae0`
  (branch `local-changes`)
- Licence: see `bergson/LICENSE`

This is **not** upstream `main`. The branch adds the functionality the reported
K-FAC results depend on:

- the fused application scheme (iii) of the appendix, in which whitening is
  folded into the projection matrices (`efficient kfac sketch`)
- a fix to K-FAC damping, so `auto_mean` scales λ by the mean eigenvalue of the
  approximation
- a batch-size cap for EK-FAC and uncompressed gradients

Removed from the copy: `.git/`, `.venv/`, `runs/`, `data/`, `benchmarks/`,
`.github/`, caches. The library source is otherwise unmodified — including
modules this work does not use.

## `corpus_filtering/`

The syntactic filters that split the corpus into D_X and D_base, and the BEAR
fact filter.

- Vendored at commit `e0d0f80d877d12dd77d5434386e8a07d74b23a24` (branch `main`)
- The four linguistic filters are in
  `src/corpus_filtering/filters/base.py`; the factual filter is in
  `src/corpus_filtering/filters/facts.py`.

**Modification:** the upstream repository is missing
`src/corpus_filtering/__init__.py` and `src/corpus_filtering/filters/__init__.py`
(only their compiled `__pycache__` entries were committed), so the package
cannot be imported as checked out. Both files have been added here, empty.

One comment in `filters/facts.py` was also corrected: it pointed at a CSV that
is not part of this bundle. The code it documents — a curated country/capital/
demonym table used as a fallback source of entity surface forms alongside the
Wikidata lookups — is unchanged. Nothing else was modified.

## `goldfish/`

The model-training recipe, from the Goldfish project (Chang et al., 2024),
which supplies the tokenizer and hyperparameter choices calibrated for small
data budgets.

Only the parts this work actually invokes are included:

| Path | Role |
|---|---|
| `lm_code/run_transformer_language_modeling.py` | the training entry point |
| `lm_code/dataset_classes.py`, `lm_code/lm_utils.py` | its imports |
| `tokenize_dataset.py` | text → token-id sequences |
| `tokenizer/` | SentencePiece unigram tokenizer, 51,200 types, trained on 130MB of English |
| `gpt_base_config.json` | upstream GPT-2 small config, kept for reference |

Note that `gpt_base_config.json` carries `vocab_size: 1` as a placeholder — the
training script sets the real vocabulary size from the tokenizer. The
architecture configs this bundle actually trains from are
`config/model/gpt2_small.json` and `config/model/gpt2_medium.json`, which state
the resulting 51,200 explicitly and match the released checkpoints on every
architecture key.

This code targets an older `transformers` release (4.35) than the attribution
stack, so it runs from its own environment — see `config/env.sh`
(`GOLDFISH_PYTHON`, `GOLDFISH_TORCHRUN`) and README §Setup.
