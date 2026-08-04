# Under the Influence — of What?

Code for *A Controlled Evaluation of Influence Functions*.

The question is whether influence functions (IFs) recover the training data
responsible for a property a model has learned. Answering it takes three steps,
and the code is organised around them:

1. **Isolate X.** Filter the training corpus into `D_X` (the sentences carrying
   the target property) and `D_base` (everything else).
2. **Verify X is learned.** Train a *full* model `f_θ` on `D_base ∪ D_X` and a
   *No-X* model `f_θ⁻` on `D_base` alone, and check that `f_θ` prefers the
   grammatical / factually correct member of each held-out minimal pair while
   `f_θ⁻` does not.
3. **Verify IFs recover X.** Score every training sentence by the influence
   contrast `ΔI(z) = I(s⁺, z) − I(s⁻, z)` and measure how much of the top-k
   falls inside `D_X`.

Two settings instantiate this: a **linguistic** one, where X is a grammatical
phenomenon, and a **factual** one, where X is a world-knowledge fact.

---

## Layout

```
run/          entry points — one per pipeline step, sbatch-able and bash-able
config/       all paths, budgets, phenomena, model architectures, hyperparameters
src/          the influence_on_what package
vendor/       bergson, corpus_filtering, goldfish (see vendor/VENDORED.md)
data/         BLiMP and BEAR, the NPI minimal-pair set, BEAR corpus statistics,
              corpus download scripts (see data/README.md)
```

Every script in `run/` opens with a comment block stating what it does, what it
writes, and why each design choice was made. Those blocks are the reference for
how a result is computed — read the one you are about to run.

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # creates .venv and installs, including the vendored deps
```

Then edit **`config/env.sh`** — it is the only file with machine-specific
paths. Set `PROJECT_ROOT` to your workspace (corpora, models and results live
there; it is not the code checkout) and check the corpus roots beneath it.
Every value can also be overridden from the environment without editing:

```bash
PROJECT_ROOT=/my/workspace bash run/train_full.sh --corpus common_corpus --budget 68M
```

**Two environments are needed.** The Goldfish training script targets
`transformers` 4.35, older than the attribution stack requires, so it runs from
its own interpreter. Point `GOLDFISH_PYTHON` and `GOLDFISH_TORCHRUN` in
`config/env.sh` at an environment built from
`vendor/goldfish/lm_code/`'s requirements. Only `run/train_*.sh` use it;
everything else uses `.venv`.

**Check before you launch.** Every `run/` script accepts `DRY_RUN=1`, which
resolves and prints all input paths (flagging any that are missing), prints
every command it would execute, and exits without writing anything:

```bash
DRY_RUN=1 bash run/train_full.sh --corpus common_corpus --budget 5.6B
```

## Running the pipeline

Naming: **budgets** are `68M`, `130M`, `1.3B`, `5.6B` (Common Corpus) and
`4.8B` (Wikipedia), each naming the number of *training tokens*.
**Phenomena** are `binding_reflexives`, `existential_there`, `wh_islands`,
`npi`, and `facts`. **Methods** are `gradsim`, `trackstar`, `kfac`, `bm25`.

### Linguistic setting

```bash
# Step 1 — filter (job array over corpus chunks; one pass does all four phenomena)
sbatch run/filter_linguistic.sh

# Step 2 — one full model per budget, four No-X models per budget
sbatch run/train_full.sh --corpus common_corpus --budget 5.6B
sbatch run/train_no_x.sh --corpus common_corpus --budget 5.6B --phenomenon binding_reflexives

# Step 2 — verify: run on both models and compare
sbatch run/evaluate.sh --corpus common_corpus --budget 5.6B
sbatch run/evaluate.sh --corpus common_corpus --budget 5.6B --phenomenon binding_reflexives

# Step 3 — attribute
sbatch run/attribute_full.sh --corpus common_corpus --budget 5.6B --phenomenon binding_reflexives
```

Repeat for each of the four budgets × four phenomena to reproduce the full grid.

### Factual setting

The fact filter needs to know which facts the full model already answers, so
the full model is trained and evaluated *before* filtering:

```bash
sbatch run/train_full.sh --corpus wikipedia
sbatch run/evaluate.sh   --corpus wikipedia

# corpus co-occurrence counts, then the filter
python -m influence_on_what.prepare.bear_cooccurrence --data-root <corpus> --models wikipedia
bash   run/filter_facts.sh --corpus wikipedia --prefetch     # alias cache, once
sbatch run/filter_facts.sh --corpus wikipedia

sbatch run/train_no_x.sh     --corpus wikipedia --phenomenon facts
sbatch run/evaluate.sh       --corpus wikipedia --phenomenon facts
sbatch run/attribute_full.sh --corpus wikipedia --phenomenon facts
```

The same sequence with `--corpus common_corpus --budget 5.6B` produces the
Common Corpus factual results.

### Baselines and appendix experiments

```bash
bash   run/evaluate.sh --model gpt2                        # GPT-2 small baseline
bash   run/evaluate.sh --model openai-community/gpt2-medium
sbatch run/attribute_no_x.sh  --budget 5.6B --phenomenon binding_reflexives  # RQ2 control
sbatch run/ablation_kfac.sh   --budget 5.6B --phenomenon all --correlate     # appendix ablation
```

## What each step produces

| Artifact | Produced by | Contents |
|---|---|---|
| `results_<phenomenon>.json` | `run/evaluate.sh` | Accuracy on every scoring metric, overall and per BLiMP file; per BEAR relation for facts. Run on both `f_θ` and `f_θ⁻`, the pair is the Step-2 verification. |
| `pairs_<phenomenon>.parquet` | `run/evaluate.sh` | One row per minimal pair: both scores, the verdict, and the margin between them. Read by attribution to select queries; the margins allow re-running any analysis under a stricter notion of "learned" than mere preference. |
| `BearFacts/stats.json` | `run/filter_facts.sh` | Sentences removed per chunk — how large the intervention on the corpus actually was. |
| `delta_I_stats.npz` | `run/attribute_full.sh` | ΔI over `D_X` and over the `D_base` sample, with score statistics. The aggregate question: is the contrast systematically higher on `D_X`? |
| `prec_at_k.json` | `run/attribute_full.sh` | Precision@k for the contrast and for each member alone, split by query verdict. The per-query question: how much of the top-k is `D_X`? |
| `top_inspect.parquet` | `run/attribute_full.sh`, `run/attribute_no_x.sh` | The top-k retrieved sentences with their text — what to read when the ranking is wrong and you want to know what was retrieved instead. |
| `results_<run_id>.json` | `run/attribute_full.sh` (analyze) | Aggregated ΔI statistics, the one-sided Wilcoxon test, and a PASS/PARTIAL/FAIL verdict. |
| `method_correlation.json` | `run/ablation_kfac.sh --correlate` | Pairwise Spearman/Pearson agreement between attribution methods on the same queries. |
| `codecarbon/<step>/emissions_<timestamp>.csv` | every `python -m influence_on_what` run | Energy and CO₂ per invocation. One file per run, so resubmissions accumulate rather than overwrite. |

These files are the inputs to every figure and table; the plotting code is not
part of this bundle. Their schemas are documented in the module docstrings of
`src/influence_on_what/attribute.py` and `analyze.py`.

## Data

Both probing benchmarks ship with the bundle — nothing is downloaded at run
time, and `data/README.md` records source, revision and license for each:

- **BLiMP** — `data/blimp/`, the 11 paradigm files the pipeline scores (CC BY 4.0,
  [github.com/alexwarstadt/blimp](https://github.com/alexwarstadt/blimp)).
  Files for other paradigms are downloaded into the same directory on first
  use, so adding a phenomenon needs no manual fetch.
- **BEAR** — `data/lm-pub-quiz/datasets/bear/`, all 60 relations / 7,731
  instances, at the revision `lm-pub-quiz` pins (CC BY-SA 4.0,
  [github.com/lm-pub-quiz/BEAR](https://github.com/lm-pub-quiz/BEAR)). Laid out
  as the `lm-pub-quiz` cache; `config/env.sh` sets `LM_PUB_QUIZ_CACHE_ROOT` so
  `Dataset.from_name("BEAR")` resolves to it. Export that variable yourself if
  you call a module outside `run/`, or let the package download its own copy.

Neither corpus is redistributed here.

- **Common Corpus** (English subset) — a collection of uncopyrighted and
  permissively licensed text from public sources.
- **English Wikipedia** (2023-11-01 dump, CC-BY-SA) —
  `data/prepare/setup_wikipedia_data.py` downloads and segments it into the
  `chunk_XX/train_full.txt` layout the pipeline expects. Run it locally and
  transfer the result; see the script header.

Corpora must be sentence-segmented (NLTK `sent_tokenize`) and laid out as
`chunk_XX/train_full.txt`, one sentence per line, before anything else runs.
Set `$COMMON_CORPUS_ROOT` and `$WIKI_ROOT` in `config/env.sh` to point at them.

The linguistic filters additionally need dependency parses of the Common Corpus
chunks, as CoNLL-U pickles at `$COMMON_CORPUS_PICKLES/chunk_XX.pkl`. Ours come from
MaChAmp v0.4.2 with default hyperparameters, trained on the multi-domain GUM
corpus as a single multi-task model (word segmentation, UPOS, XPOS,
lemmatization, morphological labelling, dependency parsing). We trained four
parsers, on `deberta-v3-large`, `luke-large`, `roberta-large` and
`ModernBERT-large`, and picked the deberta one after manually inspecting 20
sentences on which at least two of them produced different dependency
structures.

Also included in `data/`:

- `minimal_pairs_npi.tsv` — the NPI minimal-pair set constructed for this work,
  derived from Universal Dependencies English treebanks. Used instead of
  BLiMP's NPI suite.
- `bear_corpus_stats.json`, `bear_corpus_stats_wikipedia.json` — per-fact
  entity occurrence and co-occurrence counts, the basis of the
  "corpus-supported" (≥10 co-occurrences) criterion. Regenerate with
  `python -m influence_on_what.prepare.bear_cooccurrence`.
- `wikidata_alias_cache.json` — cached Wikidata labels, aliases and demonyms,
  so the fact filter does not re-query the API.

