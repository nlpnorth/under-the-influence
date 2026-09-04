# =============================================================================
# budgets.sh — token budgets and the training hyperparameters tied to them.
#
# A budget is named by the number of TOKENS OF THE GOLDFISH TOKENIZER the model
# is trained on.  Which chunks make up that budget is resolved two ways:
#
#   • from $COMMON_CORPUS_CHUNKS/manifest.json when run/prepare_corpus.sh built
#     the corpus — the shortest prefix of chunks reaching $BUDGET_TOKENS.
#   • from the fixed $BUDGET_CHUNKS lists below otherwise, reproducing the local
#     setup the experiment exactly.
#
# Architectures: GPT-2 small up to 1.3B tokens, GPT-2 medium above it.
#
#   budget       │ chunks (local) │ architecture │ eff. batch │ seed
#   ─────────────┼────────────────┼──────────────┼────────────┼──────
#   68M          │   1            │ GPT-2 small  │     64     │  43
#   130M         │   2            │ GPT-2 small  │     64     │  43
#   1.3B         │  16            │ GPT-2 small  │     64     │  43
#   5.6B         │  54 (all)      │ GPT-2 medium │     32     │ 112
#   4.8B (wiki)  │ all            │ GPT-2 medium │     32     │ 112
#

ALL_BUDGETS=(68M 130M 1.3B 5.6B)

# Chunk lists for the corpus in $COMMON_CORPUS_ROOT. 
# Locally, chunk_10 and chunk_17 are
# absent from it, hence the gap:
#
# The list below is 16 chunks and reproduces the 1.3B corpus
_CHUNKS_16="00 01 02 03 04 05 06 07 08 09 11 12 13 14 15 16"

# Budget targets in CoNLL-U tokens, for the manifest-driven path.
#
# The budget names count Goldfish-tokenizer subwords; a manifest counts CoNLL-U
# tokens, which is all a parsed corpus contains.  The bridge between them is a
# constant: 5.6044 bytes of sentence text per CoNLL-U token, measured
# over 8.9M tokens of deduplicated corpus. We note, that this is not fully
# accurate, but provides a close enough approximation.
#
# Two corpora are being counted here and they differ by the 90/10 split, so the
# distinction has to be kept explicit:
#
#   TRAINED-ON  the train_full.txt the model actually saw — the 90% split
#               run/filter_linguistic.sh writes.
#   PARSED      the chunk_XX.conllu run/prepare_corpus.sh writes, before any
#               split.  This is what manifest.json counts.
#
# The measured column below is the byte size of the TRAINED-ON corpus of each
# model, so it has to be divided by the 0.90 train ratio before it can be
# compared against a manifest.  (Getting this wrong costs 10% of every budget:
# a prefix of chunks holding 4.838B parsed tokens yields only 4.354B after the
# split, i.e. ~5.1B subwords where the 5.6B budget wants 5.66B.)
#
#   budget │ trained-on bytes │ trained-on tok │ parsed tok │ units of 68M
#   ───────┼──────────────────┼────────────────┼────────────┼──────────────
#   68M    │      331,365,837 │     59,126,015 │ 65,695,572 │  1.000
#   130M   │      663,271,416 │    118,348,336 │131,498,151 │  2.002
#   1.3B   │    6,515,639,965 │  1,162,593,670 │  1.292e9   │ 19.663
#   5.6B   │   27,243,948,420 │  4,861,171,298 │  5.401e9   │ 82.217
#
# TOKENS_PER_CHUNK below is the parsed size of one 68M unit, rounded to
# 65,700,000, and each target is a whole multiple of it so that a budget lands
# on a chunk boundary exactly.  Taking the raw figures instead would make the
# 130M budget need 2.002 chunks and so select three, overshooting by half.
#
# run/prepare_corpus.sh defaults --tokens-per-chunk to this same value; change
# one and change the other, or budgets stop landing on boundaries.
TOKENS_PER_CHUNK_DEFAULT=65700000

_TOKENS_68M=65700000            # 1 chunk
_TOKENS_130M=131400000          # 2 chunks
_TOKENS_1_3B=1314000000         # 20 chunks
_TOKENS_5_6B=5387400000         # 82 chunks

budget_config() {
    # Sets: BUDGET_CHUNKS BUDGET_TOKENS BUDGET_ARCH BUDGET_MODEL_CONFIG
    #       BUDGET_BATCH_SIZE BUDGET_BATCH_PER_DEVICE BUDGET_GRAD_ACCUM
    #       BUDGET_SEED BUDGET_TAG
    local budget="$1"
    case "$budget" in
        68M)
            BUDGET_CHUNKS="00"; BUDGET_TOKENS=$_TOKENS_68M
            BUDGET_ARCH=small;  BUDGET_BATCH_SIZE=64; BUDGET_BATCH_PER_DEVICE=32
            BUDGET_GRAD_ACCUM=2; BUDGET_SEED=43;  BUDGET_TAG=50m ;;
        130M)
            BUDGET_CHUNKS="00 01"; BUDGET_TOKENS=$_TOKENS_130M
            BUDGET_ARCH=small;  BUDGET_BATCH_SIZE=64; BUDGET_BATCH_PER_DEVICE=64
            BUDGET_GRAD_ACCUM=1; BUDGET_SEED=43;  BUDGET_TAG=100m ;;
        1.3B)
            BUDGET_CHUNKS="$_CHUNKS_16"; BUDGET_TOKENS=$_TOKENS_1_3B
            BUDGET_ARCH=small;  BUDGET_BATCH_SIZE=64; BUDGET_BATCH_PER_DEVICE=64
            BUDGET_GRAD_ACCUM=1; BUDGET_SEED=43;  BUDGET_TAG=1b ;;
        5.6B)
            BUDGET_CHUNKS=auto          # every chunk_* found under the corpus root
            BUDGET_TOKENS=$_TOKENS_5_6B
            BUDGET_ARCH=medium; BUDGET_BATCH_SIZE=32; BUDGET_BATCH_PER_DEVICE=16
            BUDGET_GRAD_ACCUM=2; BUDGET_SEED=112; BUDGET_TAG=10b ;;
        4.8B)
            # Wikipedia only — the corpus is a fixed dump, not a size-cut subset,
            # so there is no token target to select a prefix of chunks with.
            BUDGET_CHUNKS=auto; BUDGET_TOKENS=""
            BUDGET_ARCH=medium; BUDGET_BATCH_SIZE=32; BUDGET_BATCH_PER_DEVICE=16
            BUDGET_GRAD_ACCUM=2; BUDGET_SEED=112; BUDGET_TAG=wikipedia ;;
        *)
            echo "ERROR: unknown budget '$budget'. Valid: ${ALL_BUDGETS[*]} 4.8B" >&2
            return 1 ;;
    esac
    BUDGET_MODEL_CONFIG="$BUNDLE_ROOT/config/model/gpt2_${BUDGET_ARCH}.json"
}

# Select the shortest prefix of chunks whose cumulative token count reaches
# $BUDGET_TOKENS, from the manifest run/prepare_corpus.sh wrote.  Fails (and
# says why) when there is no manifest, when the budget has no token target, or
# when the corpus is too small for the budget — the last of these is a real
# error worth stopping on, since the alternative is training on less data than
# the budget names.
chunks_from_manifest() {
    local manifest="${COMMON_CORPUS_CHUNKS:-}/manifest.json"
    [[ -n "${BUDGET_TOKENS:-}" && -f "$manifest" ]] || return 1

    local py="${VENV_PYTHON:-}"
    [[ -x "$py" ]] || py="$(command -v python3 || true)"
    [[ -n "$py" ]] || return 1

    "$py" - "$manifest" "$BUDGET_TOKENS" <<'PY'
import json, re, sys

manifest, target = sys.argv[1], int(sys.argv[2])
chunks = json.load(open(manifest))["chunks"]

total, picked = 0, []
for chunk in chunks:
    picked.append(re.search(r"chunk_(\d+)", chunk["name"]).group(1))
    total += chunk["tokens"]
    if total >= target:
        break

if total < target:
    # Exit 3: the caller falls back to the fixed chunk lists when there
    # simply is no manifest, but a manifest that is too small for the budget is
    # an error to stop on, not one to route around.
    print(
        f"ERROR: {manifest} holds {total:,} tokens in {len(chunks)} chunks, "
        f"but this budget needs {target:,}. Re-run run/prepare_corpus.sh with a "
        f"larger --target-tokens.",
        file=sys.stderr,
    )
    sys.exit(3)
print(" ".join(picked))
PY
}

# Resolve BUDGET_CHUNKS against the manifest when there is one, else against a
# corpus root (BUDGET_CHUNKS=auto) or the fixed list.
# Returns the empty string when the corpus root does not exist yet, so that a
# dry run can still report the resolved paths instead of aborting.
resolve_chunks() {
    local corpus_root="$1" from_manifest status
    from_manifest="$(chunks_from_manifest)" && status=0 || status=$?
    if [[ $status -eq 0 ]]; then
        echo "$from_manifest"
        return 0
    fi
    # 3 means "manifest present but too small"; chunks_from_manifest has already
    # explained it on stderr, and falling back would quietly train on less data
    # than the budget names.
    if [[ $status -eq 3 ]]; then
        return 3
    fi
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
