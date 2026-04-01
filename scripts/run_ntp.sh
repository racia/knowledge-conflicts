#!/usr/bin/env bash
#
#SBATCH --job-name=domain_ntp
#SBATCH --output=domain_ntp_out
#SBATCH --error=domain_ntp_err
#SBATCH --partition=students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# SBATCH --nodelist=gpu09
# SBATCH --mem-per-gpu=10G
#SBATCH --mem=24G
#SBATCH --mail-user=ivakhnenko@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

cd "$HOME/knowledge-conflicts" || exit 1

source ~/.bashrc 2>/dev/null
echo "Activating conda env..."
conda activate kc1

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"


## domain adaptation on conflicting data
## same goes for baseline data
## please make sure to adjust the paths

CUDA_VISIBLE_DEVICES=0 python3 domain_adaptation/run.py \
    --model_name_or_path google/bigbird-roberta-base \
    --mlm_probability 0.0 \
    --train_file new_data/train-conflicts.csv \
    --validation_file new_data/dev.csv \
    --output_dir model_hub/bigbird-roberta-base-conflicts \
    --learning_rate 3e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
     --max_seq_length 2048 \
      --num_train_epochs 8 \
      --dataloader_num_workers 4 \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 2 \
       --gradient_accumulation_steps 2 \
         --save_strategy "epoch" \
         --logging_steps 200 \
         --seed 42 \
           --evaluation_strategy "epoch" \
           --do_train=True \
            --do_eval=True \
             --report_to "none" \
             --overwrite_output_dir

if [ $? -eq 0 ]; then
    echo "Python script domain_adaptation/run.py executed successfully."
else
    echo "Error: Python script domain_adaptation/run.py failed."
    return 1 2>/dev/null || exit 1
fi

echo "Data cleaning job completed."
conda deactivate