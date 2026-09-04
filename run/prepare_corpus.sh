#!/bin/bash
#SBATCH --job-name=iow_prepare_corpus
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
# =============================================================================
# Step 0 — deduplicate the released parse and cut it into chunks.
#
# This is where the pipeline starts.  Its input is the released CoNLL-U parse of
# Common Corpus; its output is the chunk_XX.conllu files that
# run/filter_linguistic.sh reads. 
#
# WHAT IT DOES
#   One streaming pass: a sentence is kept the first time its token forms are
#   seen and dropped every time after, and kept sentences are written to
#   chunk_00.conllu, chunk_01.conllu, … , each holding $TOKENS_PER_CHUNK tokens.
#   The dedup rule — md5 over the space-joined FORM column of the plain-integer-
#   id tokens — is the one the corpus behind the paper was built with.
#
# WHAT IT WRITES  (under $COMMON_CORPUS_CHUNKS)
#   chunk_XX.conllu   the deduplicated corpus  → run/filter_linguistic.sh
#   manifest.json     per-chunk sentence and token counts, and the checkpoint
#   seen_hashes.npy   the hash table, for --resume
#
# WHY IT MATTERS
#   manifest.json is what makes the corpus auditable: it states how many
#   sentences were seen, how many survived deduplication, and how many tokens
#   each chunk holds — the numbers config/budgets.sh maps budgets onto.
#
#
# USAGE
#   sbatch run/prepare_corpus.sh
#   sbatch run/prepare_corpus.sh --resume            # continue after a timeout
#   bash   run/prepare_corpus.sh --target-tokens 50000000   # a small corpus
#   DRY_RUN=1 bash run/prepare_corpus.sh
# =============================================================================

# Locate _lib.sh.  Under sbatch the script runs from a COPY in the SLURM spool
# directory, so a path relative to BASH_SOURCE does not lead back to the bundle;
# $SLURM_SUBMIT_DIR does, sbatch having been invoked from the bundle root.  The
# explicit check matters because `set -euo pipefail` lives inside _lib.sh: a
# failed source would otherwise carry on and die later on a missing function.
_IOW_LIB="$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
[[ -f "$_IOW_LIB" ]] || _IOW_LIB="${SLURM_SUBMIT_DIR:-.}/run/_lib.sh"
[[ -f "$_IOW_LIB" ]] || { echo "ERROR: cannot find run/_lib.sh — submit from the bundle root." >&2; exit 2; }
source "$_IOW_LIB"

# Both defaults come from config/budgets.sh (sourced by _lib.sh) rather than
# being repeated here: the chunk size and the largest budget target have to
# agree, or a budget stops landing on a chunk boundary.  $_TOKENS_5_6B is the
# PARSED token count of the 5.6B-subword budget, i.e. before the filter's 90/10
# split — which is the quantity this script produces.
RESUME=""
TARGET_TOKENS="${TARGET_TOKENS:-$_TOKENS_5_6B}"
TOKENS_PER_CHUNK="${TOKENS_PER_CHUNK:-$TOKENS_PER_CHUNK_DEFAULT}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)           RESUME="--resume"; shift ;;
        --target-tokens)    TARGET_TOKENS="$2"; shift 2 ;;
        --tokens-per-chunk) TOKENS_PER_CHUNK="$2"; shift 2 ;;
        --help|-h)          show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

# The released parse is either one .conllu.gz or a directory of shards; a
# directory is read in name order, which is the order the shards were cut in.
PARSED_FILES=("$COMMON_CORPUS_PARSED")
if [[ -d "$COMMON_CORPUS_PARSED" ]]; then
    # A while-read loop rather than mapfile: bash 3.2, still the default on
    # macOS, does not have mapfile (see the note in run/_lib.sh).
    PARSED_FILES=()
    while IFS= read -r f; do
        PARSED_FILES+=("$f")
    done < <(find "$COMMON_CORPUS_PARSED" -maxdepth 1 \
        \( -name '*.conllu' -o -name '*.conllu.gz' \) | sort)
fi

banner "Prepare corpus — deduplicate and chunk the released parse"
echo "Target tokens    : $TARGET_TOKENS"
echo "Tokens per chunk : $TOKENS_PER_CHUNK"
echo
echo "Inputs:"
for f in "${PARSED_FILES[@]}"; do check_input "parsed corpus" "$f"; done
echo
echo "Output:"
echo "  chunk directory        $COMMON_CORPUS_CHUNKS"
echo "    chunk_XX.conllu, manifest.json, seen_hashes.npy"
echo

make_dir "$COMMON_CORPUS_CHUNKS" "$SLURM_LOG_DIR"

run_cmd "$VENV_PYTHON" -m influence_on_what.prepare.corpus \
    --input "${PARSED_FILES[@]}" \
    --output-dir "$COMMON_CORPUS_CHUNKS" \
    --tokens-per-chunk "$TOKENS_PER_CHUNK" \
    --target-tokens "$TARGET_TOKENS" \
    ${RESUME:+$RESUME}

log "Done: $COMMON_CORPUS_CHUNKS"
