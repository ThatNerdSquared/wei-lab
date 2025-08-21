#!/bin/bash
#SBATCH --account=st-kwei06-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name=regen_25_test.sh
#SBATCH -o regen_25_test-%j.out
#SBATCH -e regen_25_test-%j.err
#SBATCH --mail-user=nathanyeung@outlook.com
#SBATCH --mail-type=ALL

module load gcc/7.5.0
module load python/3.12.7
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate
python3 -um src.oneoffs.rfd_w_sisters_sliding_compareregion_coc
