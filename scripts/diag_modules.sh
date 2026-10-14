#!/bin/bash
#SBATCH --account=aip-medilab
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=diag_modules
#SBATCH --output=/project/6106383/emese/prototypeProstEA/logs/diag_modules-%j.log
echo "hostname: $(hostname)"
module avail 2>&1
echo "---specific---"
module load python/3.11 2>&1 && echo "python3.11 OK"
module load StdEnv/2023 2>&1 && echo "StdEnv OK"
