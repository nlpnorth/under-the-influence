#!/bin/bash
#SBATCH --job-name=iow_evaluate
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=5:00:00
# =============================================================================
# Step 2 (c) — verify that X was learned, and that filtering removed it.
#
# Scores every held-out minimal pair under a trained checkpoint and reports how
# often the model prefers the grammatical member; for the factual setting it
# probes all 96 BEAR relations instead.  A model "learned" a query when it
# ranks the correct member first — for facts, under ALL THREE of the relation's
# templates.
#
# Run this on the full model AND on its No-X counterpart: the pair of numbers
# is the Step-2 verification.  Filtering worked when f_θ prefers s⁺ and f_θ⁻
# does not.
#
# METRICS (all reported per phenomenon)
#   full_sentence  log P(s) = Σ log p(tᵢ|t_<i)     ← the primary metric
#   length_norm    log P(s) / n_tokens
#   one_prefix     log p(diverging token | shared prefix)
#   two_prefix     log p(re-convergence token | good/bad prefix)
#   BEAR           per-relation accuracy, p_correct, rank_correct, entropy_norm
#
# The summed log-probability favours the shorter member of a pair by design.
# Length-normalizing does not remove that bias, it converts it into a
# tokenization-dependent one: the tokenizer splits "even" into two pieces while
# "only" stays one, so a normalized comparison would shift preference for
# reasons that have nothing to do with grammar.  Entities differ the same way
# ("Stockholm" vs. "Paris").  Both metrics are therefore computed in one pass
# and stored side by side; the summed variant is the primary one, matching the
# convention of the benchmarks these pairs come from.
#
# WHAT IT WRITES  (under $PROBE_DIR/<model>/)
#   results_<phenomenon>.json     accuracy on every metric, overall and per
#                                 BLiMP file; for facts, per BEAR relation
#   pairs_<phenomenon>.parquet    one row per minimal pair: both scores, the
#                                 verdict, and the margin between them (with
#                                 --pairs-output).  Two uses downstream —
#                                 attribution reads the verdicts to select
#                                 queries, and the margins allow re-running any
#                                 analysis under a stricter notion of "learned"
#                                 than mere preference.
#   bear_pairs_*.parquet          one row per (fact, template) for BEAR
#
# USAGE
#   bash run/evaluate.sh --corpus common_corpus --budget 5.6B                          # f_θ
#   bash run/evaluate.sh --corpus common_corpus --budget 5.6B --phenomenon npi         # f_θ⁻
#   bash run/evaluate.sh --model gpt2                                                  # baseline
#   bash run/evaluate.sh --model openai-community/gpt2-medium
#   DRY_RUN=1 bash run/evaluate.sh --corpus common_corpus --budget 68M
#
# By default all five target phenomena are scored (the four linguistic ones
# plus BEAR).  Restrict with --phenomena, e.g. --phenomena "quantifiers".
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CORPUS=common_corpus
BUDGET=""
PHENOMENON=""          # names the No-X model; empty = the full model
MODEL_OVERRIDE=""      # evaluate an arbitrary HF id or path (the GPT-2 baselines)
PHENOMENA="binding quantifiers island_effects manual_npi facts_bear"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)     CORPUS="$2";         shift 2 ;;
        --budget)     BUDGET="$2";         shift 2 ;;
        --phenomenon) PHENOMENON="$2";     shift 2 ;;
        --model)      MODEL_OVERRIDE="$2"; shift 2 ;;
        --phenomena)  PHENOMENA="$2";      shift 2 ;;
        --help|-h)    show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

if [[ -n "$MODEL_OVERRIDE" ]]; then
    MODEL_PATH="$MODEL_OVERRIDE"
    MODEL_NAME="$(basename "$MODEL_OVERRIDE")"
else
    [[ "$CORPUS" == wikipedia ]] && BUDGET="${BUDGET:-4.8B}"
    require_arg budget "$BUDGET"
    budget_config "$BUDGET"
    MODEL_NAME="$(model_name "$CORPUS" "$BUDGET" "$PHENOMENON")"
    MODEL_PATH="$WORK_DIR/models/$MODEL_NAME"
fi

OUT_DIR="$PROBE_DIR/$MODEL_NAME"

banner "Evaluate — $MODEL_NAME"
echo "Phenomena  : $PHENOMENA"
echo
echo "Inputs:"
check_input "model"          "$MODEL_PATH"
check_input "NPI pair set"   "$BUNDLE_ROOT/data/minimal_pairs_npi.tsv"
echo
echo "Outputs:"
echo "  results                $OUT_DIR/results_<phenomenon>.json"
echo "  per-pair scores        $OUT_DIR/pairs_<phenomenon>.parquet"
echo

make_dir "$OUT_DIR" "$SLURM_LOG_DIR"

# One invocation per phenomenon, so each gets its own per-pair parquet (which
# is what the attribution step joins against) alongside the shared results JSON.
for phen in $PHENOMENA; do
    log "=== $phen ==="
    # BEAR has no minimal pairs to dump: its per-(fact, template) scores are
    # written by probe_models.py itself.
    extra=()
    if [[ "$phen" != facts_bear ]]; then
        extra=(--pairs-output "$OUT_DIR/pairs_${phen}.parquet")
    fi
    run_cmd "$VENV_PYTHON" -m influence_on_what.probe_models \
        --model "$MODEL_PATH" \
        --phenomena "$phen" \
        --output "$OUT_DIR/results_${phen}.json" \
        ${extra[@]+"${extra[@]}"}
done

log "Done. Results under $OUT_DIR"
