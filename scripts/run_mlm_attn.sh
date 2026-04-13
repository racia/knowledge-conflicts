# cd "$HOME/knowledge-conflicts" || exit 1

## base script for domain adaptation on bidirectional LMs
## each model is saved with the exact script used for domain adaptation
## refer to domain_adaptation/report.md to see what was the setting for each model
CUDA_VISIBLE_DEVICES=0 python3 run_mlm_attn.py \
    --model_name_or_path FacebookAI/roberta-large \
    --mlm_probability 0.15 \
    --train_file original_corpus/dev.csv \
    --validation_file original_corpus/dev.csv \
    --output_dir model_hub/roberta-large-original \
    --learning_rate 3e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
     --max_seq_length 512 \
      --num_train_epochs 2 \
      --dataloader_num_workers 4 \
       --per_device_train_batch_size 16 \
       --per_device_eval_batch_size 8 \
       --gradient_accumulation_steps 2 \
         --save_strategy "epoch" \
         --logging_steps 200 \
         --seed 42 \
           --evaluation_strategy "epoch" \
           --do_train=True \
            --do_eval=True \
             --report_to "none" \
             --overwrite_output_dir