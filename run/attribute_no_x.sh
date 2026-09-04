#!/bin/bash
#SBATCH --job-name=iow_attribute_no_x
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
# =============================================================================
# Step 3 (control) — attribution against the ABLATED model f_θ⁻.
#
# Same machinery as run/attribute_full.sh, pointed at the No-X model instead.
# The model under attribution never saw D_X, so there is no D_X to retrieve:
# only D_base (train_clean) is scored, and no Prec@k is computed.  What the
# retrievals are inspected for is whether the ablated model still groups the
# phenomenon's neighbourhood together.
#
# It exists to settle an ambiguity.  When the full model retrieves items that
# are related to the phenomenon but not matched by the filter — possessive or
# accusative pronouns for a reflexive query, say — that could mean the model
# has formed an abstraction over the paradigm, or it could be plain surface
# similarity between the strings.  Running the same attribution against a model
# that never saw D_X distinguishes the two: surface similarity is a property of
# the text and survives ablation, an abstraction learned from D_X does not.
#
# Deliberately cut down relative to attribute_full.sh, since the output is read
# qualitatively rather than aggregated:
#   • K-FAC only
#   • 100 queries per phenomenon instead of the full test set
#   • top-10 inspection parquets only; no analyze step
#
# The 100 queries are drawn as a cascade over the FULL model's verdicts, so all
# three regimes are represented rather than an arbitrary slice:
#   30 retained      — full model learned it, ablated model did too
#   30 not-learned   — full model never learned it
#   40 attributable  — full model learned it, ablated model did not
# The full model is used only to assign these labels; attribution never touches it.
#
# WHAT IT WRITES  (under $ATTRIBUTION_DIR/<experiment>_no_x/seed_<seed>/attribution/)
#   top_inspect.parquet    top-10 retrieved sentences per query, with text
#   delta_I_stats.npz      ΔI over the scored D_base sample
#
# USAGE
#   bash run/attribute_no_x.sh --budget 5.6B --phenomenon binding_reflexives
#   DRY_RUN=1 bash run/attribute_no_x.sh --budget 5.6B --phenomenon npi
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

CORPUS=common_corpus
BUDGET=""
PHENOMENON=""
MAX_BASE=5000000
MAX_QUERIES=100
RETAINED_QUERIES=30
NOT_LEARNED_QUERIES=30
INSPECT_TOP_K=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --corpus)      CORPUS="$2";      shift 2 ;;
        --budget)      BUDGET="$2";      shift 2 ;;
        --phenomenon)  PHENOMENON="$2";  shift 2 ;;
        --max-base)    MAX_BASE="$2";    shift 2 ;;
        --max-queries) MAX_QUERIES="$2"; shift 2 ;;
        --arch)        ARCH="$2"; shift 2 ;;
        --help|-h)     show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

require_arg budget "$BUDGET"
require_arg phenomenon "$PHENOMENON"
if [[ "$PHENOMENON" == facts ]]; then
    echo "ERROR: this control is defined for the linguistic phenomena only." >&2
    exit 2
fi
phenomenon_config "$PHENOMENON"
budget_config "$BUDGET"
arch_config "$ARCH"

FULL_MODEL="$(model_name "$CORPUS" "$BUDGET")"
NO_X_MODEL="$(model_name "$CORPUS" "$BUDGET" "$PHENOMENON")"
FULL_MODEL_PATH="$WORK_DIR/models/$FULL_MODEL"
NO_X_MODEL_PATH="$WORK_DIR/models/$NO_X_MODEL"
EXP_NAME="$(experiment_name "$CORPUS" "$BUDGET" "$PHENOMENON" "$MAX_BASE")_no_x_q${MAX_QUERIES}"
ATTR_CONFIG="$BUNDLE_ROOT/config/attribution_linguistic.yaml"
CHUNKS="$(resolve_chunks "$COMMON_CORPUS_ROOT")"

if is_dry; then SEED="$BUDGET_SEED"
else SEED=$("$VENV_PYTHON" -c "import yaml; print(yaml.safe_load(open('$ATTR_CONFIG'))['seeds'][0])"); fi

banner "Attribution against the ABLATED model — $BUDGET / X = $PHENOMENON"
echo "Experiment : $EXP_NAME"
echo "Attributed : $NO_X_MODEL  (never trained on D_X)"
echo "Labels from: $FULL_MODEL   (verdicts only, never attributed)"
echo "Queries    : $MAX_QUERIES ($RETAINED_QUERIES retained / $NOT_LEARNED_QUERIES not-learned / rest attributable)"
echo
echo "Inputs:"
check_input "ablated model"  "$NO_X_MODEL_PATH"
check_input "full model"     "$FULL_MODEL_PATH"
check_input "corpus root"    "$COMMON_CORPUS_ROOT"
echo
echo "Output:"
echo "  top_inspect.parquet    $ATTRIBUTION_DIR/$EXP_NAME/seed_$SEED/attribution/kfac_effic_precond/"
echo

setup_scratch
is_dry || trap sync_and_cleanup EXIT
make_dir "$PROBE_DIR" "$SLURM_LOG_DIR"

# ── Verdict labels from BOTH models, needed for the cascade query selection ──
label_args=()
for role in full no_x; do
    if [[ "$role" == full ]]; then m_name="$FULL_MODEL"; m_path="$FULL_MODEL_PATH"
    else                          m_name="$NO_X_MODEL";  m_path="$NO_X_MODEL_PATH"; fi
    pairs_out="$PROBE_DIR/$m_name/pairs_${PHENOMENON}.parquet"
    if [[ -f "$pairs_out" ]]; then
        log "Per-pair labels ($role): cached → $pairs_out"
    else
        log "Scoring minimal pairs ($role, $PHENOMENON_PROBE_KEY)"
        make_dir "$(dirname "$pairs_out")"
        run_cmd "$VENV_PYTHON" -m influence_on_what.probe_models \
            --model "$m_path" --phenomena "$PHENOMENON_PROBE_KEY" --pairs-output "$pairs_out"
    fi
    if [[ "$role" == full ]]; then label_args+=(--full_pairs_parquet "$pairs_out")
    else                          label_args+=(--no_x_pairs_parquet "$pairs_out"); fi
done

# ── Prepare: D_base only — no --matched-file, because D_X was never trained on ──
log "Preparing datasets (D_base only)"
run_cmd "$VENV_PYTHON" -m influence_on_what.prepare.linguistic \
    --config "$ATTR_CONFIG" \
    --name "$EXP_NAME" \
    --model-path "$NO_X_MODEL_PATH" \
    --data-root "$COMMON_CORPUS_ROOT" \
    --chunks $CHUNKS \
    --clean-file "$PHENOMENON_FILTER_DIR/train_clean" \
    --phenomena "$PHENOMENON_PARADIGM" \
    --max-base "$MAX_BASE" \
    --max-queries "$MAX_QUERIES" \
    --retained-queries "$RETAINED_QUERIES" \
    --not-learned-queries "$NOT_LEARNED_QUERIES" \
    --artifacts-dir "$SCRATCH_DIR"

method_config kfac
token_batch_args=()
if [[ "$BUDGET_ARCH" == medium ]]; then
    token_batch_args=(--token_batch_size 16384 --trackstar_score_token_batch_size 8192)
fi

log "Attributing with K-FAC (fused scheme)"
run_cmd "$VENV_PYTHON" -m influence_on_what attribute \
    --config "$ATTR_CONFIG" \
    --name "$EXP_NAME" \
    --method "$METHOD_PY" \
    $METHOD_ARGS \
    --inspect_top_k "$INSPECT_TOP_K" \
    ${token_batch_args[@]+"${token_batch_args[@]}"} \
    --artifacts_dir "$SCRATCH_DIR" \
    ${label_args[@]+"${label_args[@]}"}

log "Done. Inspect top_inspect.parquet under $ATTRIBUTION_DIR/$EXP_NAME/"
