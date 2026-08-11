#!/bin/bash
#SBATCH --job-name=iow_ablation_kfac
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=7-00:00:00
# =============================================================================
# Ablation — how to apply (E)K-FAC, and how much the choice matters.
#
# (E)K-FAC was formulated in full parameter space: the Kronecker factors live
# in the layer dimensions and H^-1 is applied to full-dimensional gradients.
# That does not scale here, because a full-dimensional gradient would have to
# be materialised per training point at scoring time and cannot be cached for
# reuse across queries.  Three ways around it, in decreasing order of cost:
#
#   (i)   query batching   — one pass over the training data per query batch.
#                            Most faithful, by far the most expensive.
#   (ii)  sketched         — whiten each block in full parameter space, THEN
#                            project for storage.  Compatible with EK-FAC's
#                            eigenvalue correction, since whitening happens
#                            before any information is lost to the projection.
#   (iii) fused            — fold the whitening into the projection matrices,
#                            M_L = Φ_L A^-1/2 and M_R = Φ_R S^-1/2, so each
#                            block is whitened and sketched in one pair of
#                            matmuls and the full-dimensional whitened gradient
#                            is never formed.  Damping can no longer be shared
#                            between the two factors, so each is damped on its
#                            own; and EK-FAC's corrected eigenvalues no longer
#                            factorise across the sides, so (iii) is K-FAC only.
#
# Scheme (iii) is the default elsewhere in this pipeline, because it is the
# only one cheap enough to run across every phenomenon and budget.  This script
# measures what that choice costs in retrieval quality, sweeping the damping λ
# at the same time.
#
# Runs on a reduced sample of 500,000 sentences per phenomenon rather than the
# usual 5,000,000, keeping the same D_X-to-D_base proportions so the retrieval
# task stays comparable.  Scheme (i) is more expensive still than (ii) and is
# not swept here.
#
# WHAT IT WRITES
#   $ATTRIBUTION_DIR/<experiment>/seed_<seed>/attribution/<method_dir>/
#       prec_at_k.json      Precision@k per scheme and λ — the quality side of
#                           the trade-off
#       delta_I_stats.npz   the raw ΔI vectors, from which the rank correlation
#                           between two schemes on the same queries follows
#   method_correlation.json pairwise Spearman/Pearson agreement between methods
#                           (with --correlate)
#   Wall-clock for the Hessian-fitting and scoring passes is printed per method
#   — the cost side of the trade-off.
#
# USAGE
#   bash run/ablation_kfac.sh --budget 5.6B --phenomenon binding_reflexives
#   bash run/ablation_kfac.sh --budget 5.6B --phenomenon all --dampings 0.1,0.01
#   bash run/ablation_kfac.sh --budget 5.6B --phenomenon all --correlate
#   DRY_RUN=1 bash run/ablation_kfac.sh --budget 5.6B --phenomenon npi
#
#   Schemes: kfac (fused, iii) | kfac-sketched (ii) | ekfac-sketched (ii)
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

CORPUS=common_corpus
BUDGET=5.6B
PHENOMENON=all
SCHEMES="ekfac-sketched,kfac-sketched,kfac"
DAMPINGS="0.1,0.01"
MAX_BASE=500000
CORRELATE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --budget)     BUDGET="$2";     shift 2 ;;
        --phenomenon) PHENOMENON="$2"; shift 2 ;;
        --schemes)    SCHEMES="$2";    shift 2 ;;
        --dampings)   DAMPINGS="$2";   shift 2 ;;
        --max-base)   MAX_BASE="$2";   shift 2 ;;
        --correlate)  CORRELATE=1;     shift ;;
        --arch)       ARCH="$2"; shift 2 ;;
        --help|-h)    show_help "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'. See --help." >&2; exit 2 ;;
    esac
done

budget_config "$BUDGET"
arch_config "$ARCH"
ATTR_CONFIG="$BUNDLE_ROOT/config/attribution_linguistic.yaml"
CHUNKS="$(resolve_chunks "$COMMON_CORPUS_ROOT")"
FULL_MODEL="$(model_name "$CORPUS" "$BUDGET")"
FULL_MODEL_PATH="$WORK_DIR/models/$FULL_MODEL"

if [[ "$PHENOMENON" == all ]]; then PHENOMENON_LIST=("${ALL_PHENOMENA[@]}")
else PHENOMENON_LIST=("$PHENOMENON"); fi

if is_dry; then SEED="$BUDGET_SEED"
else SEED=$("$VENV_PYTHON" -c "import yaml; print(yaml.safe_load(open('$ATTR_CONFIG'))['seeds'][0])"); fi

banner "(E)K-FAC application-scheme ablation — $BUDGET"
echo "Phenomena  : ${PHENOMENON_LIST[*]}"
echo "Schemes    : $SCHEMES"
echo "Dampings λ : $DAMPINGS"
echo "D_base cap : $MAX_BASE  (reduced panel; the default elsewhere is 5000000)"
echo
echo "Inputs:"
check_input "full model"  "$FULL_MODEL_PATH"
check_input "corpus root" "$COMMON_CORPUS_ROOT"
echo

setup_scratch
is_dry || trap sync_and_cleanup EXIT

IFS=',' read -ra SCHEME_LIST  <<< "$SCHEMES"
IFS=',' read -ra DAMPING_LIST <<< "$DAMPINGS"
EXP_NAMES=()

for phenomenon in "${PHENOMENON_LIST[@]}"; do
    phenomenon_config "$phenomenon"
    EXP_NAME="ablation_${CORPUS}_${BUDGET}_${phenomenon}_base${MAX_BASE}"
    EXP_NAMES+=("$EXP_NAME")

    log "########## $phenomenon ($EXP_NAME) ##########"
    run_cmd "$VENV_PYTHON" -m influence_on_what.prepare.linguistic \
        --config "$ATTR_CONFIG" \
        --name "$EXP_NAME" \
        --model-path "$FULL_MODEL_PATH" \
        --data-root "$COMMON_CORPUS_ROOT" \
        --chunks $CHUNKS \
        --clean-file "$PHENOMENON_FILTER_DIR/train_clean" \
        --matched-file "$PHENOMENON_FILTER_DIR/train_matched" \
        --phenomena "$PHENOMENON_PARADIGM" \
        --max-base "$MAX_BASE" \
        --artifacts-dir "$SCRATCH_DIR"

    for scheme in "${SCHEME_LIST[@]}"; do
        method_config "$scheme"
        # The Kronecker factors themselves are undamped and fitted once per
        # scheme, so sweeping λ inside this loop reuses the fitted Hessian
        # instead of refitting it for every damping value.
        for damp in "${DAMPING_LIST[@]}"; do
            if [[ "$scheme" != kfac && "$scheme" != kfac-sketched && "$scheme" != ekfac-sketched ]]; then
                echo "ERROR: '$scheme' is not an (E)K-FAC application scheme." >&2; exit 2
            fi
            log "--- $phenomenon / $scheme / λ=$damp ---"
            run_cmd "$VENV_PYTHON" -m influence_on_what attribute \
                --config "$ATTR_CONFIG" \
                --name "$EXP_NAME" \
                --method "$METHOD_PY" \
                $METHOD_ARGS \
                --ekfac_lambda_damp "$damp" \
                --artifacts_dir "$SCRATCH_DIR"

            run_cmd "$VENV_PYTHON" -m influence_on_what analyze \
                --config "$ATTR_CONFIG" \
                --name "$EXP_NAME" \
                --method "$METHOD_PY" \
                $METHOD_ARGS \
                --ekfac_lambda_damp "$damp" \
                --artifacts_dir "$SCRATCH_DIR"
        done
    done
done

# ── Pairwise agreement between methods ──────────────────────────────────────
# Reads the delta_I_stats.npz files already on disk; no GPU work.
if [[ "$CORRELATE" == 1 ]]; then
    for exp_name in "${EXP_NAMES[@]}"; do
        log "--- method correlation: $exp_name ---"
        run_cmd "$VENV_PYTHON" -m influence_on_what.compare_methods \
            --artifacts-dir "$SCRATCH_DIR" \
            --name "$exp_name" \
            --seed "$SEED"
    done
fi

log "Done."
