#!/bin/bash
#SBATCH --job-name=iow_corpus_stats
#SBATCH --output=/home/jgsi/slurm_outputs/babylm_ifs/%j_iow_corpus_stats.out
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=20:00:00
#SBATCH --requeue
# =============================================================================
# Site launcher for run/corpus_stats.sh, same pattern as
# site_prepare_corpus.slurm: source site.env, cd into the bundle, run the script
# with bash so that its "source $(dirname $BASH_SOURCE)/_lib.sh" resolves (under
# a bare sbatch the script runs from the SLURM spool dir and cannot).
#
# CPU only, so no module loads and no GPU request.
#
# Statistics land beside the other results artifacts, next to the
# corpus_provenance notebook that reads them.
#
# Usage:
#   sbatch site_corpus_stats.slurm run/corpus_stats.sh --source parsed
#   sbatch site_corpus_stats.slurm run/corpus_stats.sh --source chunks
# =============================================================================
set -euo pipefail
exec 2>&1

BUNDLE=/home/jgsi/InfluenceOnWhat
set -a; source "$BUNDLE/site.env"; set +a

export COMMON_CORPUS_PARSED=/home/argy/english_data/data/cc-en-10b.conllu.gz
export COMMON_CORPUS_CHUNKS=/home/jgsi/IFsForBabyLM/data/chunks_pilot
export CORPUS_STATS_DIR=$PROJECT_ROOT/experiments/artifacts/results/corpus_stats

cd "$BUNDLE"
bash "$@"
