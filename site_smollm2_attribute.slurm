#!/bin/bash
#SBATCH --job-name=iow_smollm2_attribute
#SBATCH --output=/home/jgsi/slurm_outputs/babylm_ifs/%j_iow_smollm2_attribute.out
#SBATCH --gres=gpu:1
#SBATCH --partition=NLPNorth
#SBATCH --nodelist=cn11
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
# =============================================================================
# Site launcher for run/attribute_full.sh and run/attribute_no_x.sh — sources
# site.env and loads the modules those scripts need, same as
# site_smollm2_sweep.slurm does for training. Generic over which attribution
# script and phenomenon to run, since there's one job per (corpus, phenomenon)
# rather than a fixed task list.

#
# Usage:
#   sbatch site_smollm2_attribute.slurm run/attribute_full.sh \
#        --corpus common_corpus --budget 5.6B --phenomenon binding_reflexives --arch smollm2
# =============================================================================

set -euo pipefail
exec 2>&1

# Reduce CUDA allocator fragmentation on long streaming gradient-collection
# loops, as goldfish_attribute.slurm does. 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BUNDLE=/home/jgsi/InfluenceOnWhat
set -a; source "$BUNDLE/site.env"; set +a

module purge
module load GCCcore/12.3.0 OpenSSL/3 Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1
export PATH="$HOME/.local/bin:$PATH"

cd "$BUNDLE"
bash "$@"
