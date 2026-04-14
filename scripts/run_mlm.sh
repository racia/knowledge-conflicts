#!/usr/bin/env bash
#
# sbatch scripts/run_mlm.sh
#
#SBATCH --job-name=d_mlm
#SBATCH --output=domain_mlm
# SBATCH --error=domain_mlm
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
#export LD_LIBRARY_PATH=/home/students/ivakhnenko/miniconda3/envs/kc1/lib:$LD_LIBRARY_PATH

## domain adaptation on conflicting data
## same goes for baseline data
## please make sure to adjust the paths
SCRIPT=train_domain.py
model=google/bigbird-roberta-base # google/bigbird-roberta-base
output_dir=model_hub/bigbird-roberta-base-test
learning_rate=3e-5
epochs=8

CUDA_VISIBLE_DEVICES=0 srun python3 "$SCRIPT" \
    --model_name_or_path $model \
    --mlm_probability 0.15 \
    --train_file data/final_data/baseline_corpus/train.csv \
    --validation_file data/final_data/baseline_corpus/dev.csv \
    --output_dir $output_dir \
    --learning_rate $learning_rate \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
     --max_seq_length 2048 \
      --num_train_epochs $epochs \
      --dataloader_num_workers 4 \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 2 \
       --gradient_accumulation_steps 2 \
         --save_strategy "epoch" \
         --logging_steps 200 \
         --seed 42 \
           --eval_strategy "epoch" \
           --do_train=True \
            --do_eval=True \
             --report_to "none" \
             --overwrite_output_dir

if [ $? -eq 0 ]; then
    echo "Python script $SCRIPT executed successfully."
else
    echo "Error: Python script $SCRIPT failed."
    return 1 2>/dev/null || exit 1
fi

echo "Masked language modelling job is completed."
conda deactivate