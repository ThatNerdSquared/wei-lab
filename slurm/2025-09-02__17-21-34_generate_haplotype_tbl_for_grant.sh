#!/bin/bash
#SBATCH --account=st-kwei06-1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name=generate_haplotype_tbl_for_grant
#SBATCH -o generate_haplotype_tbl_for_grant-%j.out
#SBATCH -e generate_haplotype_tbl_for_grant-%j.err
#SBATCH --mail-user=nathanyeung@outlook.com
#SBATCH --mail-type=ALL

module load gcc/7.5.0
module load python/3.12.7
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate
python3 -um src.oneoffs.generate_haplotype_tbl_for_grant > output.backup.txt
