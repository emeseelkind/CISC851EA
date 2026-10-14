#!/bin/bash
# Retrains each held-out candidate's probe on the full outer-train split to
# get raw per-core scores (needed for an ROC operating point), same memory
# footprint as make_heatmap_examples.sh -- run via sbatch, not directly.
#
# Usage: sbatch scripts/sensitivity_at_specificity.sh \
#     --held_out_json experiments/h1/fold0/held_out_test.json \
#     --outer_fold 0 --out_dir experiments/h1/fold0 --target_specificity 0.60
#
#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=prostea_sens_spec
#SBATCH --output=logs/%x-%j.log

set -e
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDPROJ_DIR=/home/emese/projects/aip-medilab/emese/medprojects/proside-dino-pca

module load python/3.11 gcc/12.3 cuda/12.2 hdf5/1.14.6
source "$MEDPROJ_DIR/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO_DIR"
mkdir -p logs

python scripts/sensitivity_at_specificity.py --cache_dir cache/regions "$@"
