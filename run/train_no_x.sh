#!/bin/bash
#SBATCH --job-name=iow_train_no_x
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
# =============================================================================
# Step 2 (b) — train the No-X model f_θ⁻ on D_base alone.
#
# Identical to run/train_full.sh in every respect except its input corpus:
# instead of the unfiltered train_full.txt it reads train_clean.txt, the
# residue left after run/filter_linguistic.sh (or run/filter_facts.sh) has
# removed every sentence carrying the target property X.
#
# One No-X model per (corpus, budget, phenomenon) — so four per budget in the
# linguistic setting, plus one per corpus in the factual setting.
#
# The seed is fixed per budget and SHARED with the full model, so f_θ and f_θ⁻
# differ only in their training data.  Both are trained from scratch rather
# than branched from a common checkpoint: influence estimates are sensitive to
# training stochasticity, and branching would leave the two models sharing a
# trajectory that the data difference alone should not have produced.
#
# WHAT IT WRITES
#   $WORK_DIR/raw_text/<model>.txt   the filtered corpus
#   $WORK_DIR/models/<model>/        the checkpoint  ← the deliverable
#
# WHAT USES THE CHECKPOINT
#   run/evaluate.sh       gives the lower half of the Step-2 comparison: if
#                         filtering worked, accuracy drops relative to f_θ
#   run/attribute_full.sh uses its per-pair verdicts to separate queries the
#                         full model learned FROM D_X (f_θ right, f_θ⁻ wrong)
#                         from those it would have learned anyway
#
# USAGE
#   bash run/train_no_x.sh --corpus common_corpus --budget 5.6B --phenomenon binding_reflexives
#   bash run/train_no_x.sh --corpus wikipedia --phenomenon facts
#   DRY_RUN=1 bash run/train_no_x.sh --corpus common_corpus --budget 68M --phenomenon npi
#
#   Phenomena:     binding_reflexives | existential_there | wh_islands | npi | facts
#   Architectures: gpt2 (default) | smollm2
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CORPUS=common_corpus
BUDGET=""
PHENOMENON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)     CORPUS="$2";     shift 2 ;;
        --budget)     BUDGET="$2";     shift 2 ;;
        --phenomenon) PHENOMENON="$2"; shift 2 ;;
        --arch)       ARCH="$2"; shift 2 ;;
        --help|-h)    show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

require_arg phenomenon "$PHENOMENON"
phenomenon_config "$PHENOMENON"

case "$CORPUS" in
    common_corpus)
        if [[ "$PHENOMENON" == facts ]]; then
            CORPUS_ROOT="$BEAR_FILTER_ROOT"; CORPUS_FILE="BearFacts/train_clean.txt"
        else
            CORPUS_ROOT="$COMMON_CORPUS_ROOT";   CORPUS_FILE="$PHENOMENON_FILTER_DIR/train_clean.txt"
        fi
        # The held-out split lives under the linguistic filter output for both
        # phenomena, since the facts filter reads the already-split
        # train_full.txt and so writes no test.txt of its own.  It is drawn
        # with a fixed seed before any filter runs, so it is identical under
        # every phenomenon directory and is unfiltered — the No-X model is
        # evaluated on the same held-out text as its full counterpart.
        HELD_OUT_ROOT="$COMMON_CORPUS_ROOT"
        HELD_OUT_FILE="Binding-reflexive/test.txt"
        require_arg budget "$BUDGET" ;;
    wikipedia)
        if [[ "$PHENOMENON" != facts ]]; then
            echo "ERROR: the Wikipedia corpus is only used for the factual setting." >&2
            echo "       Use --phenomenon facts, or --corpus common_corpus." >&2
            exit 2
        fi
        BUDGET="${BUDGET:-4.8B}"
        CORPUS_ROOT="$BEAR_FILTER_ROOT"; CORPUS_FILE="BearFacts/train_clean.txt"
        # Wikipedia has no held-out split; train_model carves one instead.
        HELD_OUT_ROOT=""; HELD_OUT_FILE="" ;;
    *) echo "ERROR: unknown corpus '$CORPUS'. Valid: common_corpus wikipedia" >&2; exit 2 ;;
esac

budget_config "$BUDGET"
arch_config "$ARCH"
MODEL_NAME="$(model_name "$CORPUS" "$BUDGET" "$PHENOMENON")"
CHUNKS="$(resolve_chunks "$CORPUS_ROOT")"
RAW_TEXT="$WORK_DIR/raw_text/${MODEL_NAME}.txt"

banner "Train No-X model — $CORPUS / $BUDGET tokens / $ARCH $BUDGET_ARCH / X = $PHENOMENON"
echo "Model name : $MODEL_NAME"
echo "Chunks     : $(echo "$CHUNKS" | wc -w)"
echo "Seed       : $BUDGET_SEED  (shared with the full model)"
echo
echo "Inputs:"
check_input "filtered corpus"  "$CORPUS_ROOT"
check_tokenizer
check_input "model config"     "$ARCH_MODEL_CONFIG"
echo
echo "Output:"
echo "  model                  $WORK_DIR/models/$MODEL_NAME"
echo

make_dir "$SLURM_LOG_DIR"

concat_corpus "$RAW_TEXT" "$CORPUS_ROOT" "$CHUNKS" "$CORPUS_FILE"

# Shared with the full model of the same corpus and budget, hence the name.
HELD_OUT_TEXT=""
if [[ -n "$HELD_OUT_ROOT" ]]; then
    HELD_OUT_TEXT="$WORK_DIR/raw_text/${CORPUS}_${BUDGET}_heldout.txt"
    concat_corpus "$HELD_OUT_TEXT" "$HELD_OUT_ROOT" "$CHUNKS" "$HELD_OUT_FILE"
fi

train_model   "$MODEL_NAME" "$RAW_TEXT" "$HELD_OUT_TEXT"

log "Done. Next: bash run/evaluate.sh --corpus $CORPUS --budget $BUDGET --phenomenon $PHENOMENON"
