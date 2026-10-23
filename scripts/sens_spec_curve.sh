#!/bin/bash
# Same resource footprint as sensitivity_at_specificity.sh -- retrains the
# three held-out candidates' probes on the full outer-train split to get raw
# per-core scores (needed for a full ROC curve, not just one operating
# point). Run via sbatch, not directly on a login node (this job's RSS grows
# past the login node's cgroup limit and gets silently SIGKILLed).
#
# Usage: sbatch scripts/sens_spec_curve.sh \
#     --held_out_json experiments/h1_fixed/fold0/held_out_test.json \
#     --outer_fold 0 --out_dir experiments/h1_fixed/fold0
#
#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=prostea_sens_spec_curve
#SBATCH --output=logs/%x-%j.log

set -e
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDPROJ_DIR=/home/emese/projects/aip-medilab/emese/medprojects/proside-dino-pca

source "$MEDPROJ_DIR/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO_DIR"
mkdir -p logs

python scripts/sens_spec_curve.py --cache_dir cache/regions "$@"
