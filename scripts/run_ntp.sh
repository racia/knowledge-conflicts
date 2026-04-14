#!/usr/bin/env bash
#
# sbatch scripts/run_ntp.sh
#
#SBATCH --job-name=da_ntp
#SBATCH --output=da_ntp
# SBATCH --error=da_ntp
#SBATCH --partition=students
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# SBATCH --nodelist=gpu09
# SBATCH --mem-per-gpu=10G
#SBATCH --mem=48G
#SBATCH --mail-user=ivakhnenko@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

cd "$HOME/knowledge-conflicts" || exit 1

source ~/.bashrc 2>/dev/null
env=torch261
echo "Activating conda env $env..."
conda activate $env

export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-}
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"
export LD_LIBRARY_PATH="/home/students/ivakhnenko/miniconda3/envs/$env/lib:$LD_LIBRARY_PATH"

## domain adaptation on conflicting data
## same goes for baseline data
## please make sure to adjust the paths
SCRIPT=train_domain.py

# done:
# openai-community/gpt2       - original_corpus
# state-spaces/mamba-1.4b-hf  - original_corpus
# openai-community/gpt2       - baseline_corpus
# state-spaces/mamba-1.4b-hf  - baseline_corpus

models=("openai-community/gpt2" "state-spaces/mamba-1.4b-hf")
data_paths=("original_corpus" "baseline_corpus") # "conflict_corpus"

for data_path in "${data_paths[@]}"; do
  echo "Processing data path: $data_path"
  train_file="data/final_data/$data_path/train.csv"
  for model in "${models[@]}"; do
    echo "Training model: $model on data path: $train_file"
    # +   openai-community/gpt2         2048
    # +   state-spaces/mamba-1.4b-hf    OOM even with max_seq_length=64: todo: try with smaller batch size
    # - RWKV/RWKV7-Goose-World3-1.5B-HF
    # - moonshotai/Kimi-Linear-48B-A3B-Base
    output_dir="model_hub/$model-$data_path-test"
    learning_rate=3e-5
    epochs=8 # 8
    max_seq_length=2048
    per_device_train_batch_size=4

    CUDA_VISIBLE_DEVICES=0 srun python3 "$SCRIPT" \
        --model_name_or_path "$model" \
        --mlm_probability 0.0 \
        --train_file $train_file \
        --validation_file "data/final_data/$data_path/dev.csv" \
        --output_dir "$output_dir" \
        --learning_rate "$learning_rate" \
        --warmup_ratio 0.05 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
         --max_seq_length $max_seq_length \
          --num_train_epochs $epochs \
          --dataloader_num_workers 4 \
           --per_device_train_batch_size $per_device_train_batch_size \
           --per_device_eval_batch_size 1 \
           --gradient_accumulation_steps 2 \
           --gradient_checkpointing=True \
           --torch_empty_cache_steps=100 \
           --fp16=True \
             --save_strategy "epoch" \
             --logging_steps 200 \
             --seed 42 \
               --eval_strategy "epoch" \
               --do_train=True \
                --do_eval=True \
                 --report_to "none" \
                 --overwrite_output_dir
  done
done

if [ $? -eq 0 ]; then
    echo "Python script $SCRIPT executed successfully."
else
    echo "Error: Python script $SCRIPT failed."
    return 1 2>/dev/null || exit 1
fi

echo "Next tokens prediction language modelling job is completed."
conda deactivate