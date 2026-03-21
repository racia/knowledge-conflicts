from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from torch.amp import autocast
import torch

from utils.prompts.examples import FORMATTED_EXAMPLES


def load_model_tokenizer(model_name: str) -> tuple:
    """
    Load a language model and its tokenizer.

    :param model_name: Name of the model to load.
                       Supported:
                       - "meta-llama/Meta-Llama-3-8B-Instruct",
                       - "meta-llama/Llama-3.3-70B-Instruct".
    :return: A tuple containing the model and tokenizer.
    """
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is not available. A GPU is required to load the model.")

    available_models = (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
        "gemini-2.5-flash-lite",
    )
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' is not supported. Available models: {available_models}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "quantization_config": quantization_config,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
        "offload_folder": "offload_folder",
        "offload_state_dict": True,
        "offload_buffers": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    print("Tokenizer loaded successfully.")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    print("Model loaded successfully.")
    model.eval()
    torch.cuda.empty_cache()
    return model, tokenizer


def prepare_prompt(task_type: str, tokenizer):
    """
    Prepare the prompt for question formatting by combining the formatted examples.

    :param task_type: The type of task for which to prepare the prompt
                      (e.g., "question", "explanation").
    :param tokenizer: The tokenizer to use for encoding the prompt.
    :return: A string containing the formatted examples to be used in the prompt.
    """
    path = Path(f"utils/prompts/cleaning/{task_type}.txt")
    with open(path, "r", encoding="utf-8") as f:
        base_prompt = f.read()
    joined_examples = "\n".join(FORMATTED_EXAMPLES.get(task_type, []))
    base_prompt.replace("<EXAMPLES>", joined_examples)
    prompt_inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": base_prompt}],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return prompt_inputs


def check_task(fn):
    def inner_checker(*args, **kwargs):
        task = kwargs.get("task")
        required_keys = set()
        if task == "question":
            required_keys = {"question", "answer_options"}
        elif task == "explanation":
            required_keys = {"explanation"}
        if not required_keys.issubset(set(kwargs.keys())):
            raise ValueError(f"Missing required keys for task '{task}'. Required keys: {required_keys}")
        return fn(*args, **kwargs)
    return inner_checker


def _concat_token_blocks(first: dict, second: dict) -> dict:
    """
    Concatenate two tokenizer output dicts (e.g. system + user).
    Handles common keys like input_ids, attention_mask, token_type_ids if present.
    :param first: The first tokenizer output dict.
    :param second: The second tokenizer output dict.
    :return: A new dict with concatenated values for common keys and preserved unique keys.
    """
    out = {}
    for k in set(first.keys()).union(second.keys()):
        if k in first and k in second:
            out[k] = torch.cat([first[k], second[k]], dim=-1)
        elif k in first:
            out[k] = first[k]
        else:
            out[k] = second[k]
    return out


@check_task
def format_chat(system_inputs: dict, tokenizer, **kwargs) -> dict:
    """
    Format the chat messages for the model.

    :param system_inputs: The tokenized system prompt inputs.
    :param tokenizer: The tokenizer to use for encoding the messages.
    :param task: The specific task to perform ("question" or "explanation").
    :param explanation: The explanation string (required for "explanation" task).
    :param question: The original question string (required for "question" task).
    :param answer_options: A list of answer option strings (required for "question" task).
    :return: A list of formatted chat messages.
    """
    answer_options = "\n- ".join(kwargs.get("answer_options", []))
    task_map = {
        "question": f"Question: {kwargs.get('question', '')}\nAnswer options:\n{answer_options}" + " Take a deep breath and return only the formatted question: ",
        "explanation": f"Explanation: {kwargs.get('explanation', '')}" + " Take a deep breath and return only the formatted explanation: ",
        "classification": f"Paragraph: {kwargs.get('explanation', '')}" + " Take a deep breath and return only 'true' or 'false': ",
    }
    user_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": task_map[kwargs.get("task")]}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # concatenate system and user token blocks
    inputs = _concat_token_blocks(system_inputs, user_inputs)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    # clamp indices to vocab size, otherwise the model will error out with "index out of range in self"
    # input_ids = torch.clamp(input_ids, 0, tokenizer.vocab_size - 1)
    # pad input_ids and attention_mask to the same length
    # attention_mask = torch.nn.utils.rnn.pad_sequence(
    #     attention_mask,
    #     batch_first=True,
    #     padding_value=0
    # )
    torch.cuda.empty_cache()
    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def generate_text(
        model,
        tokenizer,
        inputs: dict,
        temperature: float = 0.1,
) -> str:
    """
    Generate text using the provided model and tokenizer.
    :param model: The language model to use for generation.
    :param tokenizer: The tokenizer corresponding to the model.
    :param chat_messages: A list of chat messages formatted for the model.
    :param temperature: Sampling temperature for generation.
    :return: The generated text.
    """
    with torch.no_grad():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        # allow generation of up to 20% more tokens than the input length
        max_len = int(input_len+input_len * 0.2)
        with autocast("cuda"):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_len,
                temperature=temperature
            )
        output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    torch.cuda.empty_cache()
    return output