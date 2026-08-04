#!/bin/bash
#SBATCH --job-name=iow_filter_ling
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-06:00:00
#SBATCH --array=0-55
# =============================================================================
# Step 1 (a) — isolate X in the training data, for the four linguistic phenomena.
#
# Applies a syntactic filter over the dependency-parsed corpus and splits each
# chunk into the sentences that exhibit the phenomenon (D_X) and those that do
# not (D_base).  All four filters run in a single pass over the chunk.
#
# The filters are defined in
#   vendor/corpus_filtering/src/corpus_filtering/filters/base.py
# as ExistentialThereQuantifierFilter, BindingReflexive,
# InterrogativeWhModifierFilter and NukeNPI.
#
# PREREQUISITE
#   The corpus must already be sentence-segmented (NLTK sent_tokenize) and
#   dependency-parsed.  The parses come from MaChAmp v0.4.2 trained on the
#   multi-domain GUM corpus with deberta-v3-large as backbone; the parsed
#   chunks are pickles at $COMMON_CORPUS_PICKLES/chunk_XX.pkl.
#
# WHAT IT WRITES  (per chunk, per phenomenon, under $COMMON_CORPUS_ROOT/chunk_XX/)
#   <Phenomenon>/train_full.txt      the unfiltered corpus     → run/train_full.sh
#   <Phenomenon>/train_matched.txt   D_X, the sentences with X → the retrieval target
#   <Phenomenon>/train_clean.txt     D_base, everything else   → run/train_no_x.sh
#
# WHY IT MATTERS
#   train_matched.txt IS the set whose recovery Precision@k measures, so the
#   filter defines the ground truth for the whole linguistic setting.  A filter
#   that misses instances of the phenomenon leaves them in D_base, where a
#   correct retrieval is then scored as a miss.
#
# USAGE
#   sbatch run/filter_linguistic.sh                  # all 56 chunks as a job array
#   bash   run/filter_linguistic.sh --chunk 00       # one chunk, locally
#   DRY_RUN=1 bash run/filter_linguistic.sh --chunk 00
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CHUNK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --chunk)   CHUNK="$2"; shift 2 ;;
        --help|-h) show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

# Under SLURM the array task index selects the chunk.
if [[ -z "$CHUNK" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CHUNK=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
fi
require_arg chunk "$CHUNK"

INPUT="$COMMON_CORPUS_PICKLES/chunk_${CHUNK}.pkl"
OUTPUT="$COMMON_CORPUS_ROOT/chunk_${CHUNK}/"

banner "Linguistic corpus filter — chunk_${CHUNK}"
echo "Inputs:"
check_input "parsed chunk" "$INPUT"
check_input "filter code"  "$CORPUS_FILTERING_DIR/scripts/run_all_filters.py"
echo
echo "Output:"
echo "  filtered chunk         $OUTPUT"
echo "    Binding-reflexive/           {train_full,train_matched,train_clean}.txt"
echo "    existential-there-quantifier/  \""
echo "    interrogative-wh-modifier/     \""
echo "    nuke-npi/                      \""
echo

make_dir "$OUTPUT" "$SLURM_LOG_DIR"

run_cmd "$VENV_PYTHON" "$CORPUS_FILTERING_DIR/scripts/run_all_filters.py" \
    --input "$INPUT" \
    --output "$OUTPUT"

log "Done: chunk_${CHUNK}"
