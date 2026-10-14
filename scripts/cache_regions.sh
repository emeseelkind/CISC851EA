#!/bin/bash
# Build the full-patch-grid region cache (needs the medAI env for the frozen
# PNF+ encoder -- same env ../prototypeProstate uses to build its own cache).
#
# Smoke test:
#   sbatch --time=00:15:00 scripts/cache_regions.sh --limit 8
# Full cache build (~1118 cores, the only slow step -- run once):
#   sbatch scripts/cache_regions.sh
#
#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=prostea_cache_regions
#SBATCH --output=logs/%x-%j.log

set -e
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDPROJ_DIR=/home/emese/projects/aip-medilab/emese/medprojects/proside-dino-pca

module load python/3.11 gcc/12.3 cuda/12.2 hdf5/1.14.6
source "$MEDPROJ_DIR/.venv/bin/activate"

# proside-dino-pca's medAI is the only place the PNF+ model factory
# (prostnfound_plus_final) and the OPTIMUM item adapter
# (_ProstNFoundDatasetAdapterOptimum) live.
export PYTHONPATH="$MEDPROJ_DIR:$PYTHONPATH"

cd "$REPO_DIR"
mkdir -p logs cache

python scripts/cache_regions.py \
    --out_dir cache/regions \
    --skip_existing \
    "$@"
