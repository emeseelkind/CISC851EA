#!/bin/bash
# H3 ablation: zero w_negative / w_unlabeled on each fold's evolved_best
# genome and re-evaluate on that fold's held-out test split. Same venv/CPU
# setup as scripts/run_search.sh -- see its header comment for why.
#
# Usage: sbatch scripts/ablation_h3.sh --h1_dir experiments/h1_fixed \
#     --out experiments/h1_fixed/h3_ablation.csv
#
#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=prostea_ablation_h3
#SBATCH --output=logs/%x-%j.log

set -e
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDPROJ_DIR=/home/emese/projects/aip-medilab/emese/medprojects/proside-dino-pca
source "$MEDPROJ_DIR/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$REPO_DIR"
mkdir -p logs

python scripts/ablation_h3.py --cache_dir cache/regions "$@"
