#!/bin/bash
#SBATCH --job-name=iow_corpus_stats
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=0-16:00:00
# =============================================================================
# Corpus provenance statistics — what the training data actually is.
#
# Counts the per-document licensing and provenance metadata Common Corpus
# attaches and the parse preserves (# newdoc id, # meta::…), plus sentence and
# token totals.  Produces the JSON behind the appendix's corpus composition
# figure and statistics table.
#
# WHAT IT DOES
#   Run it twice, once per --source, and compare the two outputs:
#
#     --source parsed   the released cc-en-10b.conllu.gz — every document that
#                       was parsed, before anything was thrown away
#     --source chunks   the chunk_XX.conllu run/prepare_corpus.sh wrote — the
#                       documents that survive deduplication and the token
#                       budget, i.e. what the models are trained on
#
# WHY BOTH
#   Deduplication is not metadata-blind.  A collection whose documents share
#   boilerplate loses more of itself than one of prose, so the composition of
#   the corpus the models see is not automatically the composition of the corpus
#   that was parsed, and reporting only one of the two asserts something that
#   was never checked.  The filter's later 90/10 train/held-out split IS blind —
#   an independent Bernoulli draw that never inspects a document — so it needs
#   no third pass, and statistics over the chunks describe the training split.
#
#   The parsed pass also settles a question the original corpus left open: it
#   reports how many sentences and tokens cc-en-10b.conllu.gz actually holds,
#   and so whether the historical 56-chunk build exhausted the file or merely
#   ran out of job-array slots.
#
# WHAT IT WRITES  (under $CORPUS_STATS_DIR)
#   corpus_stats_parsed.json   the full released corpus
#   corpus_stats_chunks.json   the deduplicated, budgeted subset
#
# WHAT IT COSTS
#   A read-only streaming pass that touches only comment lines and the first
#   field of each token row — cheaper per sentence than run/prepare_corpus.sh,
#   which hashes every sentence and writes it back out.  The parsed pass reads
#   ~90 GB of gzip; --cpus-per-task 2 leaves a core for pigz beside the reader.
#
# USAGE
#   sbatch run/corpus_stats.sh --source parsed
#   sbatch run/corpus_stats.sh --source chunks
#   DRY_RUN=1 bash run/corpus_stats.sh --source parsed
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

SOURCE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)  SOURCE="$2"; shift 2 ;;
        --help|-h) show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done
require_arg source "$SOURCE"

# A while-read loop rather than mapfile: bash 3.2, still the default on macOS,
# does not have mapfile (see the note in run/_lib.sh).
INPUTS=()
case "$SOURCE" in
    parsed)
        if [[ -d "$COMMON_CORPUS_PARSED" ]]; then
            while IFS= read -r f; do INPUTS+=("$f"); done < <(
                find "$COMMON_CORPUS_PARSED" -maxdepth 1 \
                    \( -name '*.conllu' -o -name '*.conllu.gz' \) | sort)
        else
            INPUTS=("$COMMON_CORPUS_PARSED")
        fi ;;
    chunks)
        while IFS= read -r f; do INPUTS+=("$f"); done < <(
            find "$COMMON_CORPUS_CHUNKS" -maxdepth 1 -name 'chunk_*.conllu' | sort) ;;
    *)
        echo "ERROR: --source must be 'parsed' or 'chunks', not '$SOURCE'." >&2
        exit 2 ;;
esac

if [[ ${#INPUTS[@]} -eq 0 ]]; then
    echo "ERROR: no CoNLL-U files found for --source $SOURCE." >&2
    exit 2
fi

OUTPUT="$CORPUS_STATS_DIR/corpus_stats_${SOURCE}.json"

banner "Corpus statistics — $SOURCE"
echo "Inputs (${#INPUTS[@]}):"
for f in "${INPUTS[@]}"; do check_input "corpus" "$f"; done
echo
echo "Output:"
echo "  statistics JSON        $OUTPUT"
echo

make_dir "$CORPUS_STATS_DIR" "$SLURM_LOG_DIR"

run_cmd "$VENV_PYTHON" -m influence_on_what.prepare.corpus_stats \
    --input "${INPUTS[@]}" \
    --output "$OUTPUT" \
    --label "$SOURCE"

log "Done: $OUTPUT"
