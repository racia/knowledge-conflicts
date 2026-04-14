import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "token": None,
        "trust_remote_code": True,  # TO CHANGE
    }
# model_name = "RWKV/v6-Finch-1B6-HF"
model_name = "RWKV/v5-Eagle-7B-HF"
# 'RWKV/RWKV7-Goose-World3-1.5B-HF' -- too experimental?
model_args = {
    "model_name_or_path": model_name,
    "model_type": None,
    "config_overrides": None,
    "config_name": None,
    "tokenizer_name": None,
    "cache_dir": None,
    "use_fast_tokenizer": True,
    "model_revision": 'main',
    "token": None,
    "trust_remote_code": True,  # TO CHANGE
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
    trust_remote_code=True,
    torch_dtype=model_args["torch_dtype"],
    low_cpu_mem_usage=model_args["low_cpu_mem_usage"],
    # quantization_config=quantization_config,  # the source of problems!
    ignore_mismatched_sizes=True,
    device_map="auto",
).to(device)
# model = model.cuda()
print("Loaded model:", model)

tokenizer = AutoTokenizer.from_pretrained(model_args["model_name_or_path"], trust_remote_code=True)
print("Loaded tokenizer:", tokenizer)

inputs = tokenizer("The sun is bright today, ", return_tensors="pt").to(device)
output = model.generate(inputs["input_ids"], max_new_tokens=333, do_sample=True, temperature=1.0, top_p=0.3, top_k=0, )

print("Finished")