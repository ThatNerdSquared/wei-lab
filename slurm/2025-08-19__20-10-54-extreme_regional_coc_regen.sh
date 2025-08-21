#!/bin/bash
#SBATCH --account=st-kwei06-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=extreme_regional_coc_regen
#SBATCH -o extreme_regional_coc_regen-%j.out
#SBATCH -e extreme_regional_coc_regen-%j.err
#SBATCH --mail-user=nathanyeung@outlook.com
#SBATCH --mail-type=ALL

module load gcc/7.5.0
module load python/3.12.7
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate
python3 -um src.oneoffs.rfd_w_sisters_expanding_window_coc_extreme_regional
