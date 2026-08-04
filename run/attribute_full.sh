#!/bin/bash
#SBATCH --job-name=iow_attribute
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4-00:00:00
# =============================================================================
# Step 3 — influence attribution against the FULL model f_θ.
#
# For every held-out minimal pair (s⁺, s⁻) this scores every training sentence
# z by the influence contrast  ΔI(z) = I(s⁺, z) − I(s⁻, z)  and asks how much
# of the top-k falls inside D_X.  Runs one method after another so they share
# the prepared dataset and (for K-FAC) the fitted Hessian.
#
# WHAT IT DOES
#   1. prepare  — build the query set (BLiMP paradigm, or the manual NPI set)
#                 and the scored training set: all of D_X plus a sample of
#                 D_base drawn at the full-corpus ratio, capped at --max-base
#   2. label    — score the pairs under BOTH f_θ and f_θ⁻ so every query can be
#                 marked learned / not-learned.  Only queries the full model
#                 got right AND the No-X model got wrong are attributable to X;
#                 the rest would be explained by something other than D_X.
#   3. attribute— per method: collect gradients, fit the preconditioner, score
#   4. analyze  — aggregate ΔI, compute Prec@k and the D_X vs D_base contrast
#
# WHAT IT WRITES  (under $ATTRIBUTION_DIR/<experiment>/seed_<seed>/attribution/<method>/)
#   delta_I_stats.npz     ΔI over D_X and over the D_base sample, with score
#                         statistics.  Answers the aggregate question: is the
#                         contrast systematically higher on D_X than on D_base?
#   prec_at_k.json        Precision@k for each ranking (the contrast, and each
#                         member alone), split by query verdict.  Answers the
#                         per-query question: how much of the top-k is D_X?
#   top_inspect.parquet   the top-k retrieved sentences with their text — the
#                         input to any qualitative reading of WHAT was retrieved
#                         when the ranking is wrong
#   results_<run>.json    aggregated statistics and a PASS/PARTIAL/FAIL verdict
#   ...and, one level up, pairs_<phenomenon>.parquet under $PROBE_DIR/<model>/:
#                         per-pair verdicts for f_θ and f_θ⁻
#
# COST
#   The dominant term is gradient collection over --max-base sentences.  At the
#   default 5,000,000 that is a few hours per method per panel on one H100.
#   Start with a much smaller --max-base to smoke-test the pipeline.
#
# USAGE
#   bash run/attribute_full.sh --corpus common_corpus --budget 5.6B --phenomenon binding_reflexives
#   bash run/attribute_full.sh --corpus wikipedia --phenomenon facts
#   bash run/attribute_full.sh --corpus common_corpus --budget 68M --phenomenon npi \
#        --methods kfac --max-base 20000
#   DRY_RUN=1 bash run/attribute_full.sh --corpus common_corpus --budget 1.3B --phenomenon wh_islands
#
#   Methods: gradsim | trackstar | kfac | bm25   (comma-separated; default all four)
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CORPUS=common_corpus
BUDGET=""
PHENOMENON=""
METHODS="gradsim,trackstar,kfac,bm25"
MAX_BASE=5000000
FORCE_REEVAL="${FORCE_REEVAL:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)     CORPUS="$2";     shift 2 ;;
        --budget)     BUDGET="$2";     shift 2 ;;
        --phenomenon) PHENOMENON="$2"; shift 2 ;;
        --methods)    METHODS="$2";    shift 2 ;;
        --max-base)   MAX_BASE="$2";   shift 2 ;;
        --help|-h)    show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

require_arg phenomenon "$PHENOMENON"
phenomenon_config "$PHENOMENON"
[[ "$CORPUS" == wikipedia ]] && BUDGET="${BUDGET:-4.8B}"
require_arg budget "$BUDGET"
budget_config "$BUDGET"

FULL_MODEL="$(model_name "$CORPUS" "$BUDGET")"
NO_X_MODEL="$(model_name "$CORPUS" "$BUDGET" "$PHENOMENON")"
FULL_MODEL_PATH="$WORK_DIR/models/$FULL_MODEL"
NO_X_MODEL_PATH="$WORK_DIR/models/$NO_X_MODEL"
EXP_NAME="$(experiment_name "$CORPUS" "$BUDGET" "$PHENOMENON" "$MAX_BASE")"

if [[ "$PHENOMENON" == facts ]]; then
    ATTR_CONFIG="$BUNDLE_ROOT/config/attribution_facts.yaml"
    CORPUS_ROOT="$BEAR_FILTER_ROOT"
else
    ATTR_CONFIG="$BUNDLE_ROOT/config/attribution_linguistic.yaml"
    CORPUS_ROOT="$COMMON_CORPUS_ROOT"
fi
CHUNKS="$(resolve_chunks "$CORPUS_ROOT")"

if is_dry; then
    SEED="$BUDGET_SEED"
else
    SEED=$("$VENV_PYTHON" -c "import yaml,sys; print(yaml.safe_load(open('$ATTR_CONFIG'))['seeds'][0])")
fi

banner "Attribution — $CORPUS / $BUDGET / X = $PHENOMENON"
echo "Experiment : $EXP_NAME"
echo "Methods    : $METHODS"
echo "D_base cap : $MAX_BASE"
echo "Seed       : $SEED"
echo
echo "Inputs:"
check_input "full model f_θ"   "$FULL_MODEL_PATH"
check_input "no-X model f_θ⁻"  "$NO_X_MODEL_PATH"
check_input "corpus root"      "$CORPUS_ROOT"
check_input "attribution cfg"  "$ATTR_CONFIG"
if [[ "$PHENOMENON" != facts ]]; then
    check_input "D_X (chunk_00)" "$CORPUS_ROOT/chunk_00/$PHENOMENON_FILTER_DIR/train_matched.txt"
fi
echo
echo "Outputs:"
echo "  attribution artifacts  $ATTRIBUTION_DIR/$EXP_NAME/seed_$SEED/attribution/<method>/"
echo "  per-pair labels        $PROBE_DIR/{$FULL_MODEL,$NO_X_MODEL}/pairs_${PHENOMENON}.parquet"
echo

setup_scratch
is_dry || trap sync_and_cleanup EXIT
make_dir "$SLURM_LOG_DIR" "$PROBE_DIR"

# ── 1. Prepare queries + training data ───────────────────────────────────────
log "Preparing datasets"
if [[ "$PHENOMENON" == facts ]]; then
    run_cmd "$VENV_PYTHON" -m influence_on_what.prepare.facts \
        --config "$ATTR_CONFIG" \
        --name "$EXP_NAME" \
        --model-path "$FULL_MODEL_PATH" \
        --bear-facts-dir "$BEAR_FILTER_ROOT" \
        --chunks $CHUNKS \
        --corpus-stats "$BUNDLE_ROOT/data/bear_corpus_stats.json" \
        --alias-cache "$BUNDLE_ROOT/data/wikidata_alias_cache.json" \
        --max-base "$MAX_BASE" \
        --artifacts-dir "$SCRATCH_DIR"
else
    run_cmd "$VENV_PYTHON" -m influence_on_what.prepare.linguistic \
        --config "$ATTR_CONFIG" \
        --name "$EXP_NAME" \
        --model-path "$FULL_MODEL_PATH" \
        --data-root "$CORPUS_ROOT" \
        --chunks $CHUNKS \
        --clean-file "$PHENOMENON_FILTER_DIR/train_clean" \
        --matched-file "$PHENOMENON_FILTER_DIR/train_matched" \
        --phenomena "$PHENOMENON_PARADIGM" \
        --max-base "$MAX_BASE" \
        --artifacts-dir "$SCRATCH_DIR"
fi

# ── 2. Per-pair learned / not-learned labels ─────────────────────────────────
# Forward passes only.  Cached on durable storage so resubmitting the job does
# not re-score a (model, phenomenon) pair that has already been done; set
# FORCE_REEVAL=1 to recompute.
label_args=()
for role in full no_x; do
    if [[ "$role" == full ]]; then m_name="$FULL_MODEL"; m_path="$FULL_MODEL_PATH"
    else                          m_name="$NO_X_MODEL";  m_path="$NO_X_MODEL_PATH"; fi

    pairs_out="$PROBE_DIR/$m_name/pairs_${PHENOMENON}.parquet"
    if [[ "$role" == no_x && ! -d "$m_path" ]]; then
        log "WARNING: No-X model missing at $m_path — proceeding without its labels."
        log "         Queries cannot be split into learned vs. not-learned; Prec@k"
        log "         will cover all queries rather than the attributable subset."
        continue
    fi
    if [[ "$FORCE_REEVAL" != "1" && -f "$pairs_out" ]]; then
        log "Per-pair labels ($role): cached → $pairs_out"
    else
        log "Scoring minimal pairs ($role, $PHENOMENON_PROBE_KEY)"
        make_dir "$(dirname "$pairs_out")"
        run_cmd "$VENV_PYTHON" -m influence_on_what.probe_models \
            --model "$m_path" \
            --phenomena "$PHENOMENON_PROBE_KEY" \
            --pairs-output "$pairs_out"
    fi
    if [[ "$role" == full ]]; then label_args+=(--full_pairs_parquet "$pairs_out")
    else                          label_args+=(--no_x_pairs_parquet "$pairs_out"); fi
done

# ── 3 + 4. Attribute and analyze, one method at a time ───────────────────────
# GPT-2 medium on the large corpora leaves little headroom after the model
# weights and TrackStar buffers, so the token batches are halved there.
token_batch_args=()
if [[ "$BUDGET_ARCH" == medium ]]; then
    token_batch_args=(--token_batch_size 16384 --trackstar_score_token_batch_size 8192)
fi

IFS=',' read -ra METHOD_LIST <<< "$METHODS"
for method in "${METHOD_LIST[@]}"; do
    method_config "$method"
    log "=== $method (python method: $METHOD_PY, output dir: $METHOD_DIR) ==="

    run_cmd "$VENV_PYTHON" -m influence_on_what attribute \
        --config "$ATTR_CONFIG" \
        --name "$EXP_NAME" \
        --method "$METHOD_PY" \
        $METHOD_ARGS \
        ${token_batch_args[@]+"${token_batch_args[@]}"} \
        --artifacts_dir "$SCRATCH_DIR" \
        ${label_args[@]+"${label_args[@]}"}

    run_cmd "$VENV_PYTHON" -m influence_on_what analyze \
        --config "$ATTR_CONFIG" \
        --name "$EXP_NAME" \
        --method "$METHOD_PY" \
        $METHOD_ARGS \
        --artifacts_dir "$SCRATCH_DIR" \
        ${label_args[@]+"${label_args[@]}"}

    # Sync this method's results out of node-local scratch as soon as it is
    # done, so a later failure cannot lose them.  query_index/ is a large
    # regenerable intermediate and is not kept.
    if ! is_dry; then
        dest="$ATTRIBUTION_DIR/$EXP_NAME/seed_$SEED/attribution/$METHOD_DIR"
        mkdir -p "$dest"
        rsync -a --timeout=120 --exclude="query_index/" \
            "$SCRATCH_DIR/$EXP_NAME/seed_$SEED/attribution/$METHOD_DIR/" "$dest/"
        rm -rf "$SCRATCH_DIR/$EXP_NAME/seed_$SEED/attribution/$METHOD_DIR"
        log "Synced → $dest"
    fi
done

log "Done. Results under $ATTRIBUTION_DIR/$EXP_NAME/seed_$SEED/attribution/"
