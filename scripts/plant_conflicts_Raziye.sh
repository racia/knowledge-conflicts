#!/usr/bin/env bash
#
#SBATCH --job-name=plant_q
#SBATCH --output=planting_out-%j
#SBATCH --error=planting_err-%j
# SBATCH --partition=students
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
# SBATCH --nodelist=gpu09
# SBATCH --mem-per-gpu=10G
#SBATCH --mem=64G
#SBATCH --mail-user=sari@cl.uni-heidelberg.de
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
echo "Activating conda env..."
conda activate kc1


export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

SCRIPT="planting/conflict_planting.py"

splits=(train)  # possible values: dev, test, train
source="cleaned" # classical: "original/clean_spaces_id", for explanations: "cleaned"

declare -a CONFIGS=("$PWD/configs/confl_plant_1.yaml")
#task="question"  # question, classification, explanation

echo "Running conflict planting with the following configurations: ${CONFIGS[*]}"
srun python "$SCRIPT" --splits "${splits[@]}" --source "$source" --config "${CONFIGS[@]}"

if [ $? -eq 0 ]; then
    echo "Python script conflict_planting.py executed successfully."
else
    echo "Error: Python script conflict_planting.py failed."
    return 1 2>/dev/null || exit 1
fi

echo "Conflict planting job completed."
conda deactivate
