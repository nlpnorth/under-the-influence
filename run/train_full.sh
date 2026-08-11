#!/bin/bash
#SBATCH --job-name=iow_train_full
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
# =============================================================================
# Step 2 (a) — train the FULL model f_θ on D_train = D_base ∪ D_X.
#
# There is exactly ONE full model per (corpus, budget): it sees the entire
# corpus, so it is shared across all four linguistic phenomena and the factual
# setting.  The per-phenomenon counterparts are trained by run/train_no_x.sh.
#
# WHAT IT DOES
#   1. concatenates the unfiltered corpus (chunk_XX/…/train_full.txt)
#   2. tokenizes to sequences of at most 512 tokens (Goldfish SentencePiece
#      unigram tokenizer, 51,200 types)
#   3. shuffles at the sequence level, holds out 2,000 sequences
#   4. trains for exactly one epoch from scratch
#
# WHAT IT WRITES
#   $WORK_DIR/raw_text/<model>.txt              assembled plain-text corpus
#   $WORK_DIR/tokenized_data/<model>.txt        token-id sequences
#   $WORK_DIR/tokenized_data_split/<model>.txt  train split (+ _eval2k.txt)
#   $WORK_DIR/models/<model>/                   the checkpoint  ← the deliverable
#
# WHAT USES THE CHECKPOINT
#   run/evaluate.sh       scores it on the held-out minimal pairs, giving the
#                         upper half of the Step-2 comparison
#   run/attribute_full.sh attributes against it — this is the model whose
#                         behaviour the influence scores are meant to explain
#
# USAGE
#   bash run/train_full.sh --corpus common_corpus --budget 5.6B
#   bash run/train_full.sh --corpus wikipedia
#   bash run/train_full.sh --corpus wikipedia --arch smollm2
#   DRY_RUN=1 bash run/train_full.sh --corpus common_corpus --budget 68M
#   sbatch      run/train_full.sh --corpus common_corpus --budget 5.6B
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CORPUS=common_corpus
BUDGET=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)  CORPUS="$2"; shift 2 ;;
        --budget)  BUDGET="$2"; shift 2 ;;
        --arch)   ARCH="$2"; shift 2 ;;
        --help|-h) show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

case "$CORPUS" in
    common_corpus)
        require_arg budget "$BUDGET"
        CORPUS_ROOT="$COMMON_CORPUS_ROOT"
        # train_full.txt is filter-independent — the unfiltered corpus is the
        # same file under every phenomenon's directory, so we read it from one.
        CORPUS_FILE="Binding-reflexive/train_full.txt" ;;
    wikipedia)
        BUDGET="${BUDGET:-4.8B}"
        CORPUS_ROOT="$WIKI_ROOT"
        CORPUS_FILE="train_full.txt" ;;
    *) echo "ERROR: unknown corpus '$CORPUS'. Valid: common_corpus wikipedia" >&2; exit 2 ;;
esac

budget_config "$BUDGET"
arch_config "$ARCH"
MODEL_NAME="$(model_name "$CORPUS" "$BUDGET")"
CHUNKS="$(resolve_chunks "$CORPUS_ROOT")"
RAW_TEXT="$WORK_DIR/raw_text/${MODEL_NAME}.txt"

banner "Train FULL model — $CORPUS / $BUDGET tokens / $ARCH $BUDGET_ARCH"
echo "Model name : $MODEL_NAME"
echo "Chunks     : $(echo "$CHUNKS" | wc -w)"
echo "Seed       : $BUDGET_SEED"
echo "Batch      : $BUDGET_BATCH_SIZE ($BUDGET_BATCH_PER_DEVICE x $BUDGET_GRAD_ACCUM)"
echo
echo "Inputs:"
check_input "corpus root"   "$CORPUS_ROOT"
check_tokenizer
check_input "model config"  "$ARCH_MODEL_CONFIG"
check_input "training code" "$GOLDFISH_DIR/lm_code/run_transformer_language_modeling.py"
echo
echo "Output:"
echo "  model                  $WORK_DIR/models/$MODEL_NAME"
echo

make_dir "$SLURM_LOG_DIR"

concat_corpus "$RAW_TEXT" "$CORPUS_ROOT" "$CHUNKS" "$CORPUS_FILE"
train_model   "$MODEL_NAME" "$RAW_TEXT"

log "Done. Next: bash run/evaluate.sh --corpus $CORPUS --budget $BUDGET"
