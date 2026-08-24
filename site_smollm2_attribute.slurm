#!/bin/bash
#SBATCH --job-name=iow_smollm2_attribute
#SBATCH --output=/home/jgsi/slurm_outputs/babylm_ifs/%j_iow_smollm2_attribute.out
#SBATCH --gres=gpu:1
#SBATCH --partition=NLPNorth
#SBATCH --exclude=cn19
#SBATCH --mem=80G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
# =============================================================================
# Site launcher for run/attribute_full.sh and run/attribute_no_x.sh — sources
# site.env and loads the modules those scripts need, same as
# site_smollm2_sweep.slurm does for training. Generic over which attribution
# script and phenomenon to run, since there's one job per (corpus, phenomenon)
# rather than a fixed task list.
#
# cn19 is excluded: a training job may still be running there, and a second
# concurrent job on that node has twice caused it to crash mid-training.
#
# Usage:
#   sbatch site_smollm2_attribute.slurm run/attribute_full.sh \
#        --corpus common_corpus --budget 5.6B --phenomenon binding_reflexives --arch smollm2
# =============================================================================

set -euo pipefail
exec 2>&1

BUNDLE=/home/jgsi/InfluenceOnWhat
set -a; source "$BUNDLE/site.env"; set +a

module purge
module load GCCcore/12.3.0 OpenSSL/3 Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1
export PATH="$HOME/.local/bin:$PATH"

cd "$BUNDLE"
bash "$@"
