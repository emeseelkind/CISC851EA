#!/bin/bash
# NSGA-II loss search for one outer fold. Only touches cached embeddings
# (scripts/cache_regions.sh must have populated cache/regions first) and
# trains a single linear layer per genome -- no medAI, no GPU needed, CPU is
# fine unless pop_size/n_gen get large. Uses the proside-dino-pca venv (same
# as scripts/cache_regions.sh): the `prostNfound` conda env's numpy build is
# broken on this cluster's login node (missing libcpupower.so.0).
#
# Usage: sbatch scripts/run_search.sh --outer_fold 0 --pop_size 16 --n_gen 10
#
#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=prostea_search
#SBATCH --output=logs/%x-%j.log

set -e
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDPROJ_DIR=/home/emese/projects/aip-medilab/emese/medprojects/proside-dino-pca

# NOTE: this cluster's module catalog no longer has python/3.11, gcc/12.3,
# cuda/12.2 or hdf5/1.14.6 (confirmed via `module avail` on a compute node,
# 2026-10-05) -- they must have existed when this script was first written,
# but `module load` for any of them now fails outright, and under `set -e`
# that killed the job before it ever reached the venv activation below.
# Not needed anyway: this script is pure-Python/CPU (see header comment) and
# the venv's wheels (torch, h5py, numpy) are self-contained, so just use the
# venv directly.
source "$MEDPROJ_DIR/.venv/bin/activate"

# Match torch/OpenMP's thread pool to the allocation -- left at its default
# (one thread per visible CPU) it oversubscribes and can even fail thread
# creation entirely on a constrained (e.g. login-node) cgroup.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$REPO_DIR"
mkdir -p logs experiments

python scripts/run_search.py \
    --cache_dir cache/regions \
    --exp_dir experiments/search \
    "$@"
