import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("Using device:", device)

config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "token": None,
        "trust_remote_code": False,
    }
model_args = {
    "model_name_or_path": 'state-spaces/mamba-1.4b-hf',
    "model_type": None,
    "config_overrides": None,
    "config_name": None,
    "tokenizer_name": None,
    "cache_dir": None,
    "use_fast_tokenizer": True,
    "model_revision": 'main',
    "token": None,
    "trust_remote_code": False,
    "torch_dtype": None,
    "low_cpu_mem_usage": True,
}
config = AutoConfig.from_pretrained(model_args["model_name_or_path"], **config_kwargs)
print("Loaded config:", config)

model = AutoModelForCausalLM.from_pretrained(
    model_args["model_name_or_path"],
    from_tf=model_args["model_name_or_path"].endswith(".ckpt"),
    config=config,
    cache_dir=model_args["cache_dir"],
    revision=model_args["model_revision"],
    token=model_args["token"],
    trust_remote_code=model_args["trust_remote_code"],
    torch_dtype=model_args["torch_dtype"],
    low_cpu_mem_usage=model_args["low_cpu_mem_usage"],
    # quantization_config=quantization_config,  # the source of problems!
    ignore_mismatched_sizes=True,
    device_map="auto",
)
print("Loaded model:", model)

tokenizer = AutoTokenizer.from_pretrained(model_args["model_name_or_path"])
print("Loaded tokenizer:", tokenizer)

print("Finished")
# dataset = load_dataset("Abirate/english_quotes", split="train")
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=3,
#     per_device_train_batch_size=4,
#     logging_dir='./logs',
#     logging_steps=10,
#     learning_rate=2e-3
# )
# lora_config =  LoraConfig(
#         r=8,
#         target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
#         task_type="CAUSAL_LM",
#         bias="none"
# )
# trainer = SFTTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     args=training_args,
#     peft_config=lora_config,
#     train_dataset=dataset,
#     dataset_text_field="quote",
# )
# trainer.train()