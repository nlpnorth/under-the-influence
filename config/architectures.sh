# =============================================================================
# architectures.sh — which model architecture to train.
#
# The pipeline's claims are about what influence functions recover, not about a
# particular backbone, so the architecture is a free axis: the corpus, the
# filters, the minimal pairs and the attribution settings are all unchanged
# when it varies.  Swapping it is how you check that a result is a property of
# the method rather than of GPT-2.
#
# An architecture supplies two things — a randomly-initialised model config to
# train from, and a tokenizer.  Everything else (one epoch, the per-budget
# batch sizes and seeds in budgets.sh, sequences capped at 512 tokens) is held
# fixed across architectures.
#
#   gpt2      GPT-2 small / medium, with the SentencePiece unigram tokenizer
#             from vendor/goldfish (51,200 types, trained on 130MB of English).
#   smollm2   SmolLM2-135M / -360M — a Llama-style decoder: SwiGLU, RMSNorm,
#             RoPE, grouped-query attention, tied embeddings — with SmolLM2's
#             own BPE tokenizer (49,152 types).
#
# The size within an architecture is chosen by budget (see budgets.sh):
# BUDGET_ARCH=small below 1.3B tokens, medium above.  The two SmolLM2 sizes
# share a tokenizer, so only the config differs between them.
# =============================================================================

ALL_ARCHITECTURES=(gpt2 smollm2)

arch_config() {
    # Sets: ARCH_MODEL_CONFIG ARCH_TOKENIZER ARCH_TOKENIZER_IS_LOCAL
    # Requires BUDGET_ARCH (small|medium), so call budget_config first.
    local arch="$1"
    if [[ -z "${BUDGET_ARCH:-}" ]]; then
        echo "ERROR: arch_config needs BUDGET_ARCH — call budget_config first." >&2
        return 1
    fi
    case "$arch" in
        gpt2)
            ARCH_MODEL_CONFIG="$BUNDLE_ROOT/config/model/gpt2_${BUDGET_ARCH}.json"
            ARCH_TOKENIZER="$TOKENIZER"
            ARCH_TOKENIZER_IS_LOCAL=1 ;;
        smollm2)
            ARCH_MODEL_CONFIG="$BUNDLE_ROOT/config/model/smollm2_${BUDGET_ARCH}.json"
            case "$BUDGET_ARCH" in
                small)  ARCH_TOKENIZER="HuggingFaceTB/SmolLM2-135M" ;;
                medium) ARCH_TOKENIZER="HuggingFaceTB/SmolLM2-360M" ;;
            esac
            # Resolved from the HuggingFace hub rather than a directory, so it
            # is copied into the checkpoint with save_pretrained instead of cp.
            ARCH_TOKENIZER_IS_LOCAL=0 ;;
        *)
            echo "ERROR: unknown architecture '$arch'. Valid: ${ALL_ARCHITECTURES[*]}" >&2
            return 1 ;;
    esac
}
