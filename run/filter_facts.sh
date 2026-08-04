#!/bin/bash
#SBATCH --job-name=iow_filter_facts
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --array=0-55
# =============================================================================
# Step 1 (b) — isolate X in the training data, for the factual setting.
#
# X is a single BEAR fact (an entity–relation triplet).  For every fact that
# qualifies, this removes from the corpus:
#   • every sentence where the two entities CO-OCCUR, and
#   • every sentence merely mentioning one of them, but only for entities that
#     appear in fewer than 30,000 corpus sentences.  The cap stops the filter
#     from deleting a large, topically coherent slice of the corpus for a
#     frequent entity such as "Europe", which would confound the f_θ vs. f_θ⁻
#     comparison with a general distribution shift.
#
# A fact qualifies when it is CORPUS-SUPPORTED (its entities co-occur in at
# least 10 sentences, --min-cooccur-count) and the FULL model already answers
# it, judged from the probe results JSON.  Filtering facts the full model never
# learned would shrink the corpus without changing anything f_θ⁻ can be asked.
# Facts whose two entities share a surface form (Italy/Italian, Mexico
# City/Mexico) cannot be separated by any surface filter and are dropped by
# --exclude-alias-overlap.
#
# Entity surface forms come from the Wikidata API (labels, aliases, demonyms),
# BEAR's own alias lists, and a curated country/demonym/capital table.  The API
# lookups are cached; STEP 0 below builds that cache once, single-threaded,
# before the array tasks (which are read-only against it) start.
#
# PREREQUISITES
#   1. run/train_full.sh + run/evaluate.sh for this corpus — the probe results
#      JSON identifies which facts the full model actually learned.
#   2. corpus co-occurrence counts:
#        python -m influence_on_what.prepare.bear_cooccurrence --data-root ...
#      writing data/bear_corpus_stats.json (Common Corpus) or
#      data/bear_corpus_stats_wikipedia.json.
#
# WHAT IT WRITES  (per chunk, under $BEAR_FILTER_ROOT/chunk_XX/BearFacts/)
#   train_clean.txt                     D_base                → run/train_no_x.sh
#   facts/<fact_slug>/cooccurrence/train_matched.txt   D_X, both entities
#   facts/<fact_slug>/subj_occurrence/train_matched.txt  subject-only sentences
#   facts/<fact_slug>/obj_occurrence/train_matched.txt   object-only sentences
#   stats.json                          per-chunk counts (also the done-marker)
#   excluded_bear_facts.json            the manifest of filtered facts
#
# WHY IT MATTERS
#   The per-fact cooccurrence/ directories are the retrieval target for Prec@k
#   in the factual setting, so this filter defines the ground truth there.  Its
#   stats.json also quantifies the intervention: the removal is deliberately
#   small relative to the corpus (well under 2% of sentences), so f_θ and f_θ⁻
#   differ in fact coverage rather than in overall data volume.
#
# USAGE
#   bash   run/filter_facts.sh --corpus common_corpus --prefetch   # STEP 0, run once
#   sbatch run/filter_facts.sh --corpus common_corpus              # all chunks
#   bash   run/filter_facts.sh --corpus wikipedia --chunk 00
#   DRY_RUN=1 bash run/filter_facts.sh --corpus common_corpus --chunk 00
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CORPUS=common_corpus
CHUNK=""
PREFETCH=0
PROBE_RESULTS=""

# The full model must be reasonably confident about a fact before removing its
# evidence is informative: a fact it answers by luck would show no drop under
# filtering regardless of what the filter did.
P_CORRECT_THRESHOLD=0.1     # min probability the full model puts on the correct answer
ENTROPY_THRESHOLD=0.55      # max normalized entropy (lower = more confident)
MIN_COOCCUR_COUNT=10        # the "corpus-supported" criterion
OCCURRENCE_CAP=30000        # entity-frequency cap for single-entity removal

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)         CORPUS="$2";        shift 2 ;;
        --chunk)          CHUNK="$2";         shift 2 ;;
        --probe-results)  PROBE_RESULTS="$2"; shift 2 ;;
        --prefetch)       PREFETCH=1;         shift ;;
        --help|-h)        show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

case "$CORPUS" in
    common_corpus)
        BUDGET=5.6B
        CORPUS_STATS="$BUNDLE_ROOT/data/bear_corpus_stats.json"
        CORPUS_DATASET=10b                       # column name inside the stats JSON
        SOURCE_ROOT="$COMMON_CORPUS_ROOT"
        SOURCE_FILE="Binding-reflexive/train_full.txt" ;;
    wikipedia)
        BUDGET=4.8B
        CORPUS_STATS="$BUNDLE_ROOT/data/bear_corpus_stats_wikipedia.json"
        CORPUS_DATASET=wikipedia
        SOURCE_ROOT="$WIKI_ROOT"
        SOURCE_FILE="train_full.txt" ;;
    *) echo "ERROR: unknown corpus '$CORPUS'. Valid: common_corpus wikipedia" >&2; exit 2 ;;
esac

FULL_MODEL="$(model_name "$CORPUS" "$BUDGET")"
: "${PROBE_RESULTS:=$PROBE_DIR/$FULL_MODEL/results_facts_bear.json}"
ALIAS_CACHE="$BUNDLE_ROOT/data/wikidata_alias_cache.json"
FILTER_PY="$CORPUS_FILTERING_DIR/scripts/run_bear_facts_filter.py"

common_args=(
    --results "$PROBE_RESULTS"
    --alias-cache "$ALIAS_CACHE"
    --corpus-stats "$CORPUS_STATS"
    --corpus-dataset "$CORPUS_DATASET"
    --p-correct-threshold "$P_CORRECT_THRESHOLD"
    --entropy-threshold "$ENTROPY_THRESHOLD"
    --min-cooccur-count "$MIN_COOCCUR_COUNT"
    --exclude-alias-overlap
)

# ── STEP 0: build the Wikidata alias cache (run once, before the array) ──────
if [[ "$PREFETCH" == 1 ]]; then
    banner "BEAR facts filter — prefetch alias cache ($CORPUS)"
    check_input "probe results" "$PROBE_RESULTS"
    echo "  alias cache            $ALIAS_CACHE"
    echo
    if [[ -f "$ALIAS_CACHE" ]]; then
        log "Alias cache already exists — nothing to do."
        exit 0
    fi
    run_cmd "$VENV_PYTHON" "$FILTER_PY" "${common_args[@]}" --prefetch-only
    log "Alias cache built → $ALIAS_CACHE"
    exit 0
fi

if [[ -z "$CHUNK" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CHUNK=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
fi
require_arg chunk "$CHUNK"

INPUT="$SOURCE_ROOT/chunk_${CHUNK}/${SOURCE_FILE}"
OUTPUT="$BEAR_FILTER_ROOT/chunk_${CHUNK}/"
DONE_MARKER="$OUTPUT/BearFacts/stats.json"

banner "BEAR facts filter — $CORPUS / chunk_${CHUNK}"
echo "Thresholds : p_correct>=$P_CORRECT_THRESHOLD, entropy<$ENTROPY_THRESHOLD,"
echo "             cooccur>=$MIN_COOCCUR_COUNT, single-entity cap=$OCCURRENCE_CAP"
echo
echo "Inputs:"
check_input "corpus chunk"   "$INPUT"
check_input "probe results"  "$PROBE_RESULTS"
check_input "corpus stats"   "$CORPUS_STATS"
check_input "alias cache"    "$ALIAS_CACHE"
echo
echo "Output:"
echo "  filtered chunk         $OUTPUT"
echo

if [[ -f "$DONE_MARKER" ]]; then
    log "chunk_${CHUNK} already filtered — skipping."
    exit 0
fi

make_dir "$OUTPUT" "$SLURM_LOG_DIR"

run_cmd "$VENV_PYTHON" "$FILTER_PY" \
    --input "$INPUT" \
    --output "$OUTPUT" \
    "${common_args[@]}" \
    --excluded-facts-output "$OUTPUT/excluded_bear_facts.json" \
    --occurrence-subject-count-threshold "$OCCURRENCE_CAP" \
    --occurrence-object-count-threshold "$OCCURRENCE_CAP"

log "Done: chunk_${CHUNK}"
