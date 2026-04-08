#!/usr/bin/env bash
#
#SBATCH --job-name=plant
#SBATCH --output=planting_out
# SBATCH --error=planting_err
#SBATCH --partition=students
#SBATCH --ntasks=1
# SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu09
# SBATCH --mem-per-gpu=10G
#SBATCH --mem=24G
#SBATCH --mail-user=ivakhnenko@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

# JOB STEPS
echo "Starting conflict planting job..."
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

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-}
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
# => RuntimeError: !block->expandable_segment_ INTERNAL ASSERT FAILED at "../c10/cuda/CUDACachingAllocator.cpp":2814, please report a bug to PyTorch.
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
#export LD_LIBRARY_PATH=/home/students/ivakhnenko/miniconda3/envs/kc1/lib:$LD_LIBRARY_PATH

env=conflicts1
echo "Activating conda env '$env'..."
conda activate $env
SCRIPT="plant_conflicts.py"

splits=(train)  # possible values: dev, test, train
source="cleaned/final" # classical: "original/clean_spaces_id", for explanations: "cleaned"

declare -a CONFIGS=("$PWD/configs/confl_plant_1.yaml")
#task="question"  # question, classification, explanation

echo "Running conflict planting with the following configurations: ${CONFIGS[*]}"
srun python3 "$SCRIPT" --splits "${splits[@]}" --source "$source" --config "${CONFIGS[@]}" --reverse

if [ $? -eq 0 ]; then
    echo "Python script $SCRIPT executed successfully."
else
    echo "Error: Python script $SCRIPT failed."
    return 1 2>/dev/null || exit 1
fi

echo "Conflict planting job completed."
conda deactivate
