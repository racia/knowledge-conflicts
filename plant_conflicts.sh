#!/usr/bin/env bash
#
#SBATCH --job-name=plant_q
#SBATCH --output=planting_q_out
#SBATCH --error=planting_q_err
#SBATCH --partition=gpu_a100_short
#SBATCH --time=00:29:00
# SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# SBATCH --nodelist=gpu09
# SBATCH --mem-per-gpu=10G
#SBATCH --mem=24G
#SBATCH --mail-user=sari@cl.uni-heidelberg.de
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
conda activate kc

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

splits=(train)  # possible values: dev, test, train
source="cleaned" # classical: "original/clean_spaces_id", for explanations: "cleaned"
#task="question"  # question, classification, explanation
srun python3 conflict_planting.py --splits "${splits[@]}" --source "$source"

if [ $? -eq 0 ]; then
    echo "Python script conflict_planting.py executed successfully."
else
    echo "Error: Python script conflict_planting.py failed."
    return 1 2>/dev/null || exit 1
fi

echo "Conflict planting job completed."
conda deactivate
