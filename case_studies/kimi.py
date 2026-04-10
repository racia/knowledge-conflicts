import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
# moonshotai/Kimi-Linear-48B-A3B-Base (needs "transformers>=4.53.0,<5.0.0" due to model_utils module) -- just too huge (only 45% is loaded when using 4-bit quantization, and the rest is offloaded to CPU and RAM of 48G)
# moonshotai/Kimi-K2-Base (quantization issues): ['awq', 'bitsandbytes_4bit', 'bitsandbytes_8bit', 'gptq', 'aqlm', 'quanto', 'eetq', 'hqq', 'compressed-tensors', 'fbgemm_fp8', 'torchao']
# moonshotai/Moonlight-16B-A3B (metadata bug in the repo)
model_name = "moonshotai/Kimi-Linear-48B-A3B-Base"
quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
)

print("quantization_config", quantization_config)

torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

messages = [
    {"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
    {"role": "user", "content": "Is 123 a prime?"}
]
torch.cuda.empty_cache()
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
torch.cuda.empty_cache()
generated_ids = model.generate(inputs=input_ids, max_new_tokens=500)
torch.cuda.empty_cache()
response = tokenizer.batch_decode(generated_ids)[0]
print(response)