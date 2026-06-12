#!/bin/bash
#SBATCH --account=st-kwei06-1
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name=recombinant_drive_measure_rf_ratio
#SBATCH -o recombinant_drive_measure_rf_ratio-%j.out
#SBATCH -e recombinant_drive_measure_rf_ratio-%j.err
#SBATCH --mail-user=nathanyeung@outlook.com
#SBATCH --mail-type=ALL

module load gcc/7.5.0
module load python/3.12.7
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate
case "$SLURM_ARRAY_TASK_ID" in
    "0")
        python3 -um src.oneoffs.recombinant_drive_measure_rf_ratio
        ;;
    "1")
        python3 -um src.oneoffs.recombinant_drive_measure_rf_ratio --interf
        ;;
    *)
        echo "Invalid job index."
        ;;
esac
