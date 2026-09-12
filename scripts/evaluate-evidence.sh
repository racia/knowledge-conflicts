#!/bin/bash

#SBATCH --job-name=judge-evidence
#SBATCH --output=judge-evidence-%j.out
#SBATCH --error=judge-evidence-%j.err
#SBATCH --time=12:00:00 # ~30 minutes for 8 samples, 12.5 hours for 200 samples
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
# SBATCH --partition=dev_gpu_h100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sari@cl.uni-heidelberg.de

# JOB STEPS
echo "Starting LLM-judge evidence evaluating job..."
case "$PWD" in 
    "$HOME"/*/knowledge-conflicts|"$HOME"/knowledge-conflicts)
    ;; # Already in the correct directory, do nothing
    *)
    if ! cd "$HOME/CrossTempNLP/knowledge-conflicts"; then
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

SCRIPT="evaluating/judge-evidence.py"

#splits=(train)  # possible values: dev, test, train
#source="cleaned" # classical: "original/clean_spaces_id", for explanations: "cleaned"

#declare -a CONFIGS=("$PWD/configs/confl_plant_1.yaml")
#task="question"  # question, classification, explanation

#echo "Running conflict planting with the following configurations: ${CONFIGS[*]}"
srun python "$SCRIPT" # --splits "${splits[@]}" --source "$source" --config "${CONFIGS[@]}"

if [ $? -eq 0 ]; then
    echo "Python script judge-evidence.py executed successfully."
else
    echo "Error: Python script judge-evidence.py failed."
    return 1 2>/dev/null || exit 1
fi

echo "LLM-judge evidence evaluating job completed."
conda deactivate
