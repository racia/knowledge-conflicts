#!/usr/bin/env bash
#
#SBATCH --job-name=probeMC
#SBATCH --output=probeMC_out_%j
#SBATCH --error=probeMC_err-%j
# SBATCH --partition=students
# SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mail-user=sari@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

# JOB STEPS
echo "Starting probeMC job..."
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

# Monitor GPU usage in background
(
    while true; do
        echo "== GPU Status: $(date) =="
        nvidia-smi --query-gpu=index,utilization.memory,utilization.gpu --format=csv
        sleep 30
    done
) > gpu_monitor.log &
MONITOR_PID=$!#


source ~/.bashrc 2>/dev/null
echo "Activating conda env..."

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

conda activate kc1
SCRIPT="probing/probeMC.py"

# splits=(train)  # possible values: dev, test, train
# source="cleaned" # classical: "original/clean_spaces_id", for explanations: "cleaned"

declare -a CONFIGS=("$PWD/configs/probeMCQ.yaml") # TODO: configure
#task="question"  # question, classification, explanation

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Running without any configuration files."
    srun python3.11 "$SCRIPT"
else
    echo "Running probeMC with the following configurations: ${CONFIGS[*]}"
    srun python3.11 "$SCRIPT" --config "${CONFIGS[@]}"
fi

if [ $? -eq 0 ]; then
    echo "Python script probeMC.py executed successfully."
else
    echo "Error: Python script probeMC.py failed."
    return 1 2>/dev/null || exit 1
fi

echo "ProbeMC job completed."
conda deactivate
