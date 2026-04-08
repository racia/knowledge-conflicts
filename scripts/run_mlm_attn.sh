cd "$HOME/knowledge-conflicts" || exit 1

## base script for domain adaptation on bidirectional LMs
## each model is saved with the exact script used for domain adaptation
CUDA_VISIBLE_DEVICES=0 python3 run_mlm_attn.py \
    --model_name_or_path google/bigbird-roberta-base \
    --mlm_probability 0.15 \
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