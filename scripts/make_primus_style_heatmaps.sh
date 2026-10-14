#!/bin/bash
# Same memory profile as make_heatmap_examples.sh -- run via sbatch, not on
# the login node.
#
# Usage: sbatch scripts/make_primus_style_heatmaps.sh \
#     --genome_json experiments/search/held_out_test.json --genome_key evolved_best \
#     --out_dir experiments/search/heatmap_examples --n_examples 5 --epochs 3
#
#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=prostea_primus_style
#SBATCH --output=logs/%x-%j.log

set -e
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDPROJ_DIR=/home/emese/projects/aip-medilab/emese/medprojects/proside-dino-pca

module load python/3.11 gcc/12.3 cuda/12.2 hdf5/1.14.6
source "$MEDPROJ_DIR/.venv/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "$REPO_DIR"
mkdir -p logs

python scripts/make_primus_style_heatmaps.py --cache_dir cache/regions "$@"
