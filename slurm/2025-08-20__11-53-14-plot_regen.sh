#!/bin/bash
#SBATCH --account=st-kwei06-1
#SBATCH --array=0-5
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --job-name=plot_regen
#SBATCH -o plot-regen-%A-%a.out
#SBATCH -e plot_regen-%A-%a.err
#SBATCH --mail-user=nathanyeung@outlook.com
#SBATCH --mail-type=ALL

module load gcc/7.5.0
module load python/3.12.7
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

case "$SLURM_ARRAY_TASK_ID" in
    "0")
        python3 -um src.oneoffs.rfd_w_sisters_expanding_window_coc_extreme_regional
        ;;
    "1")
        python3 -um src.oneoffs.rfd_w_sisters_expanding_window_coc_extreme_regional --large
        ;;
    "2")
        python3 -um src.oneoffs.rfd_w_sisters_sliding_compareregion_coc
        ;;
    "3")
        python3 -um src.oneoffs.rfd_w_sisters_sliding_compareregion_coc --alt-seed --best-bin-size-only
        ;;
    "4")
        python3 -um src.oneoffs.rfd_w_sisters_sliding_compareregion_coc --large --best-bin-size-only
        ;;
    "5")
        python3 -um src.oneoffs.rfd_w_sisters_sliding_compareregion_coc --large --alt-seed --best-bin-size-only
        ;;
    *)
        echo "Invalid job index."
        ;;
esac
