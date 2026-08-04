# =============================================================================
# env.sh — every machine-specific path used by the run/ scripts.
#
# This is the ONLY file that needs editing to move the pipeline to another
# machine.  It is sourced by every script in run/; nothing else hardcodes a
# path.  The defaults below assume a SLURM cluster with per-job node-local
# /scratch.
#
# Override any value from the environment without editing this file, e.g.
#   PROJECT_ROOT=/my/checkout bash run/train_full.sh --budget 68M
# =============================================================================

# ── This bundle ──────────────────────────────────────────────────────────────
# Root of the InfluenceOnWhat checkout (the directory containing run/, src/…).
: "${BUNDLE_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ── Project workspace on the cluster ─────────────────────────────────────────
# Where corpora, filter output, trained models and results live.  This is a
# large scratch/NFS area, NOT the code checkout.
: "${PROJECT_ROOT:=/home/jgsi/InfluenceOnWhat}"

# ── Corpora ──────────────────────────────────────────────────────────────────
# Sentence-segmented, syntactically parsed Common Corpus, one dir per chunk:
#   $COMMON_CORPUS_ROOT/chunk_XX/<Phenomenon>/{train_full,train_matched,train_clean}.txt
# Produced by run/filter_linguistic.sh (see data/prepare/ for the download step).
: "${COMMON_CORPUS_ROOT:=$PROJECT_ROOT/data/filter_output}"

# Sentence-segmented English Wikipedia, one dir per chunk:
#   $WIKI_ROOT/chunk_XX/train_full.txt
# Produced locally by data/prepare/setup_wikipedia_data.py, then transferred.
: "${WIKI_ROOT:=$PROJECT_ROOT/data/wikipedia}"

# Per-fact BEAR filter output, one dir per chunk (run/filter_facts.sh):
#   $BEAR_FILTER_ROOT/chunk_XX/BearFacts/facts/<fact_slug>/{cooccurrence,subj_occurrence,obj_occurrence}/
: "${BEAR_FILTER_ROOT:=$PROJECT_ROOT/data/bear_facts_filter_output}"

# Raw parsed corpus pickles consumed by the linguistic filters.
: "${COMMON_CORPUS_PICKLES:=$PROJECT_ROOT/data/pickles}"

# ── Benchmarks ───────────────────────────────────────────────────────────────
# Both probing benchmarks ship with the bundle (see data/README.md), so the
# pipeline runs on nodes without network access.  lm-pub-quiz resolves
# Dataset.from_name("BEAR") against $LM_PUB_QUIZ_CACHE_ROOT/datasets/ and
# downloads only if that directory is missing.  BLiMP needs no variable: the
# Python default already points at data/blimp/.
: "${LM_PUB_QUIZ_CACHE_ROOT:=$BUNDLE_ROOT/data/lm-pub-quiz}"
export LM_PUB_QUIZ_CACHE_ROOT

# ── Outputs ──────────────────────────────────────────────────────────────────
# Training workspace: raw_text/, tokenized_data/, models/, …
: "${WORK_DIR:=$PROJECT_ROOT/artifacts/training}"
# Step-2 verification results (probe_models.py JSON + per-pair parquet).
: "${PROBE_DIR:=$PROJECT_ROOT/artifacts/probe}"
# Step-3 attribution artifacts (delta_I_stats.npz, prec_at_k.json, …).
: "${ATTRIBUTION_DIR:=$PROJECT_ROOT/artifacts/attribution}"
# SLURM stdout/stderr.
: "${SLURM_LOG_DIR:=$PROJECT_ROOT/slurm_outputs}"

# ── Python ───────────────────────────────────────────────────────────────────
# Interpreter for the InfluenceOnWhat package (created by `uv sync`, see README).
: "${VENV_PYTHON:=$BUNDLE_ROOT/.venv/bin/python}"
# The Goldfish training script needs its own environment: it targets an older
# transformers release (4.35) than the attribution stack.  See the README.
: "${GOLDFISH_PYTHON:=/home/argy/.conda/envs/goldfish/bin/python3}"
: "${GOLDFISH_TORCHRUN:=/home/argy/.conda/envs/goldfish/bin/torchrun}"

# ── Vendored third-party code (see vendor/VENDORED.md) ───────────────────────
: "${BERGSON_DIR:=$BUNDLE_ROOT/vendor/bergson}"
: "${CORPUS_FILTERING_DIR:=$BUNDLE_ROOT/vendor/corpus_filtering}"
: "${GOLDFISH_DIR:=$BUNDLE_ROOT/vendor/goldfish}"
# SentencePiece unigram tokenizer, 51,200 types, trained on 130MB of English
# following the Goldfish recipe.  Shared by every model, both corpora.
: "${TOKENIZER:=$GOLDFISH_DIR/tokenizer}"

# ── Node-local scratch ───────────────────────────────────────────────────────
# Gradient index builds do many random writes into large mmap files; on NFS
# this costs ~30s/batch instead of ~0.3s.  Attribution writes here and syncs
# back to $ATTRIBUTION_DIR at the end.  Falls back to $TMPDIR off-cluster.
: "${SCRATCH_ROOT:=/scratch}"

# ── SLURM defaults ───────────────────────────────────────────────────────────
: "${SLURM_PARTITION:=NLPNorth}"

# ── Dry run ──────────────────────────────────────────────────────────────────
# DRY_RUN=1 prints every command that would be executed, resolves and reports
# all input paths, and exits without running anything or writing any file.
: "${DRY_RUN:=0}"
