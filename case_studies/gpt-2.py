import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("Using device:", device)

quantization_config = BitsAndBytesConfig( # the source of problems!
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True,
)
config_kwargs = {
        "cache_dir": None,
        "revision": 'main',
        "token": None,
        "trust_remote_code": False,
    }
model_args = {
    "model_name_or_path":'openai-community/gpt2',
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
loaded_config = {
  "_name_or_path": "openai-community/gpt2",
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": None,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": False,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": False,
  "scale_attn_weights": True,
  "summary_activation": None,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": True,
  "summary_type": "cls_index",
  "summary_use_proj": True,
  "task_specific_params": {
    "text-generation": {
      "do_sample": True,
      "max_length": 50
    }
  },
  "transformers_version": "4.47.1",
  "use_cache": True,
  "vocab_size": 50257
}

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