#!/usr/bin/env bash
#
#SBATCH --job-name=clean_a
#SBATCH --output=clean_a_out
#SBATCH --error=clean_a_err
#SBATCH --partition=students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# SBATCH --nodelist=gpu08
# SBATCH --mem-per-gpu=10G
#SBATCH --mem=24G
#SBATCH --mail-user=ivakhnenko@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

# JOB STEPS
echo "Starting data cleaning job..."
case "$PWD" in 
    "$HOME"/*/knowledge-conflicts|"$HOME"/knowledge-conflicts)
    ;; # Already in the correct directory, do nothing
    *)
    if ! cd "$HOME/knowledge-conflicts"; then
        echo "Failed to change directory to $HOME/knowledge-conflicts" >&2
        return 1 2>/dev/null || exit 1
    fi
    ;;
esac

source ~/.bashrc 2>/dev/null
echo "Activating conda env..."
conda activate kc1

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

source="original/clean_spaces_id" # classical: "original/clean_spaces_id", for explanations: "cleaned"
splits=(test train)  # possible values: dev, test, train
task="answer"  # question, answer, classification, explanation
# reverse (influences whether the list of data items is reversed or not)
#filtering_ids=op_list_disappeared_ids
srun python3 clean_data.py --source "${source}" --splits "${splits[@]}" --task "${task}" # --filtering_ids "${filtering_ids}" # --reverse

if [ $? -eq 0 ]; then
    echo "Python script clean_data.py executed successfully."
else
    echo "Error: Python script clean_data.py failed."
    return 1 2>/dev/null || exit 1
fi

echo "Data cleaning job completed."
conda deactivate