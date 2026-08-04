# =============================================================================
# budgets.sh — token budgets and the training hyperparameters tied to them.
#
# A budget is named by the number of TOKENS the model is trained on.  The
# corpus chunks on disk are named by the size of the SOURCE TEXT they were cut
# from, so the two numbering schemes differ; $BUDGET_CHUNKS below is the bridge.
#
# Architecture follows the parameter-to-token ratio: GPT-2 small up to 1.3B
# tokens, GPT-2 medium above it.
#
#   budget       │ source text │ chunks │ architecture │ eff. batch │ seed
#   ─────────────┼─────────────┼────────┼──────────────┼────────────┼──────
#   68M          │   50 MB     │  1     │ GPT-2 small  │     64     │  43
#   130M         │  100 MB     │  2     │ GPT-2 small  │     64     │  43
#   1.3B         │    1 GB     │ 16     │ GPT-2 small  │     64     │  43
#   5.6B         │   10 GB     │ 56     │ GPT-2 medium │     32     │ 112
#   4.8B (wiki)  │ Wikipedia   │ all    │ GPT-2 medium │     32     │ 112
#
# Seeds are fixed per budget and shared between a full model and its No-X
# counterpart, so the two differ in their training data and nothing else.
# Every model is trained for exactly one epoch.
# =============================================================================

ALL_BUDGETS=(68M 130M 1.3B 5.6B)

# Chunk lists.  chunk_11 and chunk_17 are absent from the corpus, hence the gaps.
_CHUNKS_16="00 01 02 03 04 05 06 07 08 09 10 12 13 14 15 16"

budget_config() {
    # Sets: BUDGET_CHUNKS BUDGET_ARCH BUDGET_MODEL_CONFIG BUDGET_BATCH_SIZE
    #       BUDGET_BATCH_PER_DEVICE BUDGET_GRAD_ACCUM BUDGET_SEED BUDGET_TAG
    local budget="$1"
    case "$budget" in
        68M)
            BUDGET_CHUNKS="00"
            BUDGET_ARCH=small;  BUDGET_BATCH_SIZE=64; BUDGET_BATCH_PER_DEVICE=32
            BUDGET_GRAD_ACCUM=2; BUDGET_SEED=43;  BUDGET_TAG=50m ;;
        130M)
            BUDGET_CHUNKS="00 01"
            BUDGET_ARCH=small;  BUDGET_BATCH_SIZE=64; BUDGET_BATCH_PER_DEVICE=64
            BUDGET_GRAD_ACCUM=1; BUDGET_SEED=43;  BUDGET_TAG=100m ;;
        1.3B)
            BUDGET_CHUNKS="$_CHUNKS_16"
            BUDGET_ARCH=small;  BUDGET_BATCH_SIZE=64; BUDGET_BATCH_PER_DEVICE=64
            BUDGET_GRAD_ACCUM=1; BUDGET_SEED=43;  BUDGET_TAG=1b ;;
        5.6B)
            BUDGET_CHUNKS=auto          # every chunk_* found under the corpus root
            BUDGET_ARCH=medium; BUDGET_BATCH_SIZE=32; BUDGET_BATCH_PER_DEVICE=16
            BUDGET_GRAD_ACCUM=2; BUDGET_SEED=112; BUDGET_TAG=10b ;;
        4.8B)
            # Wikipedia only — the corpus is a fixed dump, not a size-cut subset.
            BUDGET_CHUNKS=auto
            BUDGET_ARCH=medium; BUDGET_BATCH_SIZE=32; BUDGET_BATCH_PER_DEVICE=16
            BUDGET_GRAD_ACCUM=2; BUDGET_SEED=112; BUDGET_TAG=wikipedia ;;
        *)
            echo "ERROR: unknown budget '$budget'. Valid: ${ALL_BUDGETS[*]} 4.8B" >&2
            return 1 ;;
    esac
    BUDGET_MODEL_CONFIG="$BUNDLE_ROOT/config/model/gpt2_${BUDGET_ARCH}.json"
}

# Resolve BUDGET_CHUNKS=auto against a corpus root, or echo the fixed list.
# Returns the empty string when the corpus root does not exist yet, so that a
# dry run can still report the resolved paths instead of aborting.
resolve_chunks() {
    local corpus_root="$1"
    if [[ "$BUDGET_CHUNKS" != auto ]]; then
        echo "$BUDGET_CHUNKS"
        return 0
    fi
    if [[ ! -d "$corpus_root" ]]; then
        echo ""
        return 0
    fi
    find "$corpus_root" -maxdepth 1 -name 'chunk_*' -type d 2>/dev/null \
        | sed 's|.*/chunk_||' | sort | tr '\n' ' '
}
