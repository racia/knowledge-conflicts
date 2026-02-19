#!/usr/bin/env bash
#
#SBATCH --job-name=clean_c
#SBATCH --output=cleaning_c_out
#SBATCH --error=cleaning_c_err
#SBATCH --partition=students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# SBATCH --nodelist=gpu09
# SBATCH --mem-per-gpu=10G
#SBATCH --mem=24G
#SBATCH --mail-user=ivakhnenko@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

# JOB STEPS
echo "Starting data cleaning job..."
cd ~/knowledge-conflicts || exit 1

source ~/.bashrc 2>/dev/null
conda activate kc1

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

splits=(train)  # possible values: dev, test, train
source="original/clean_spaces_id" # classical: "original/clean_spaces_id", for explanations: "cleaned"
task="classification"  # question, classification, explanation
srun python3 clean_data.py --splits "${splits[@]}" --source "$source" --task "$task"

if [ $? -eq 0 ]; then
    echo "Python script clean_data.py executed successfully."
else
    echo "Error: Python script clean_data.py failed."
    exit 1
fi

echo "Data cleaning job completed."
deactivate