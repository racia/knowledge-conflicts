cd "$HOME/knowledge-conflicts" || exit 1

CUDA_VISIBLE_DEVICES=0 python3 domain_adaptation/run_mlm.py \
    --model_name_or_path google/bigbird-roberta-base \
    --mlm_probability 0.15 \
    --train_file train-conflicts.csv \
    --validation_file dev.csv \
    --output_dir model_hub/bigbird-roberta-base-conflicted \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
     --max_seq_length 3072 \
      --num_train_epochs 5 \
      --dataloader_num_workers 4 \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 2 \
       --gradient_accumulation_steps 2 \
         --save_strategy "epoch" \
         --logging_steps 500 \
         --seed 42 \
           --evaluation_strategy "epoch" \
           --do_train=True \
            --do_eval=True \
             --report_to "none" \
             --overwrite_output_dir