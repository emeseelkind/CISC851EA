#!/bin/bash
# H2: random-search baseline at equal budget to scripts/run_search.sh. Same
# environment/profile as run_search.sh (CPU-only, cached embeddings).
#
# Usage: sbatch scripts/random_search_baseline.sh --outer_fold 0 --exp_dir experiments/h2/fold0 \
#     --pop_size 16 --n_gen 10
#
#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=prostea_random_search
#SBATCH --output=logs/%x-%j.log

set -e
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDPROJ_DIR=/home/emese/projects/aip-medilab/emese/medprojects/proside-dino-pca

module load python/3.11 gcc/12.3 cuda/12.2 hdf5/1.14.6
source "$MEDPROJ_DIR/.venv/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$REPO_DIR"
mkdir -p logs experiments

python scripts/random_search_baseline.py \
    --cache_dir cache/regions \
    --exp_dir experiments/h2 \
    "$@"
