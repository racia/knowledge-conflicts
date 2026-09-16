#!/usr/bin/env bash
#
#SBATCH --job-name=probeMC
#SBATCH --output=probeMC-conf_out-%j
#SBATCH --error=probeMC-conf_err-%j
# SBATCH --partition=students
#SBATCH --ntasks=1
# SBATCH --nodelist=gpu08
#SBATCH --mem=64G
# SBATCH --time=00:29:29 #(~15 min for 1.7k sampples * 3 models)
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

#conda activate kc #1
source .venv/bin/activate
SCRIPT="probing.probeMC"

# splits=(train)  # possible values: dev, test, train
# source="cleaned" # classical: "original/clean_spaces_id", for explanations: "cleaned"

declare -a CONFIGS=("$PWD/configs/probeMCQ-exp-conf.yaml") # !!! --- ATTENTION --- !!!: configure for default or modified settings
#task="question"  # question, classification, explanation

if [ ${#CONFIGS[@]} -eq 0 ]; then
    echo "Running without any configuration files."
    srun python3.11 "$SCRIPT"
else
    echo "Running probeMC with the following configurations: ${CONFIGS[*]}"
    srun python3 -m "$SCRIPT" --config "${CONFIGS[@]}"
fi

if [ $? -eq 0 ]; then
    echo "Python script probeMC.py executed successfully."
else
    echo "Error: Python script probeMC.py failed."
    return 1 2>/dev/null || exit 1
fi

echo "ProbeMC job completed."
#conda deactivate
