from pathlib import Path
import random
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from torch.amp import autocast
import torch
import gc
import logging

import transformers
from utils.prompts.examples import FORMATTED_EXAMPLES


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelLoader:
    data_cleaner = None

    def __init__(self, load_model: bool = True, load_tokenizer: bool = True, load_pipeline: bool = False):
        self.load_model = load_model
        self.load_tokenizer = load_tokenizer
        self.load_pipeline = load_pipeline
        

    def set_data_cleaner(self, data_cleaner):
        self.data_cleaner = data_cleaner

    def load_model_tokenizer(self, model_name: str) -> tuple:
        """
        Load a language model and its tokenizer.

        :param model_name: Name of the model to load.
                        Supported:
                        - "meta-llama/Meta-Llama-3-8B-Instruct",
                        - "meta-llama/Llama-3.3-70B-Instruct".
        :return: A tuple containing the model and tokenizer.
        """

        model, tokenizer, pipeline = None, None, None
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is not available. A GPU is required to load the model.")

        available_models = (
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct",
            "gemini-2.5-flash-lite",
            "aaditya/OpenBioLLM-Llama3-8B",
            "Qwen/Qwen2.5-7B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen3-32B"
        )
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' is not supported. Available models: {available_models}")

        if any([model in model_name.lower() for model in ["llama", "qwen", "mistral", "gemini", "openbio"]]):        
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
                # "temperature": 0.7,
                "attn_implementation": "eager",
                "low_cpu_mem_usage": True,
                "offload_folder": "offload_folder",
                "offload_state_dict": True,
                "offload_buffers": True,
            }
            gc.collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache() # Clear GPU memory before loading the model to ensure maximum available memory for the model loading process

            # Add memory tracking
            if torch.cuda.is_available():
                logging.info("GPU Memory Summary after model load:")
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    logging.info(f"Device {i} ({torch.cuda.get_device_name(i)}):\n{torch.cuda.memory_summary(device=i)}")
            else:
                logging.info("CUDA not available; running on CPU.")
        
            if "OpenBio" in model_name:
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_name,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device="cuda",
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                model.eval()

            return model, tokenizer, pipeline
        else:
            print(f"Model with name {model_name} not found.")


    def prepare_prompt(self, task_type: str, prompt_path: Path, sys_prompt: str = None, processed: dict = None, exp_str: str = None, exp_upd_str: str = None, shuffle_order: bool = False) -> str:
        """
        Prepare the prompt for question formatting by combining the formatted examples.

        :param task_type: The type of task for which to prepare the prompt
                        (e.g., "question", "explanation").
        :param tokenizer: The tokenizer to use for encoding the prompt.
        :param sys_prompt: An optional system prompt to prepend to the base prompt.
        :param processed: The processed sample dictionary containing (shuffled) sample data to format into the prompt.
        :return: A string containing the formatted examples to be used in the prompt.
        """
        if sys_prompt is None:
            sys_prompt = ""
        with open(prompt_path, "r", encoding="utf-8") as f:
            base_prompt = f.read()
        if task_type in ["question", "explanation", "classification"]:
            cleaning_path = Path(f"utils/prompts/cleaning/{task_type}.txt")
            joined_examples = "\n".join(FORMATTED_EXAMPLES.get(task_type, []))
            base_prompt = base_prompt.replace("<EXAMPLES>", joined_examples)
        elif "mcq" in task_type.lower():
            ans_opt_pattern = re.compile(r"[\w\s]+(?=[ABCD].)")
            if processed:
                # print(f"Original base prompt: {base_prompt}")
                cleaned_exp = self.data_cleaner.clean_exp_with_cop(exp_str, processed.get(
                        f"op{chr(ord('a')+processed['cop_new']-1)}", ""
                        )) if exp_str else ""
                print(f"Cop_new: {processed.get('cop_new', '')}, Cop_upd: {processed.get('cop_upd', '')}")
                cleaned_exp = self.data_cleaner.clean_exp_with_cop(exp_upd_str, processed.get(
                        f"op{chr(ord('a')+processed['cop_upd']-1)}", ""
                        )) if exp_upd_str else exp_str  # TODO: Adapt to exp_upd for later differentiation
                options = [processed.get(f"opa", ""), processed.get(f"opb", ""), processed.get(f"opc", ""), processed.get(f"opd", "")]
                option_labels = ['A', 'B', 'C', 'D']
                base_prompt = base_prompt.format(
                    question=processed.get("question_upd", ""),
                    context=cleaned_exp if (exp_str or exp_upd_str) else "", # TODO: Extend to simultaneous use of exp_str and exp_upd_str
                    opa=options[0],
                    opb=options[1],
                    opc=options[2],
                    opd=options[3],
                )
                prompt_list = base_prompt.rsplit("\\n\n", 2)
                # print(f"Prompt list after split: {prompt_list}")
                ques_cont = re.split(r".*(?=A\.)", prompt_list[1])
                intr = prompt_list[0]
                instr = prompt_list[-1]
                if shuffle_order:
                    combined = list(zip(options, option_labels))
                    random.shuffle(combined)
                    shuffled_options, shuffled_labels = zip(*combined)
                    lab_A = shuffled_labels.index('A') # TODO: LAter for token bias
                    first_lab = shuffled_labels[0]
                    last_lab = shuffled_labels[-1]
                    base_prompt = "\n".join((intr, ques_cont[0], "\n".join([f"{label}. {option}" for label, option in zip(shuffled_labels, shuffled_options)]), instr))
                else:
                    base_prompt = "\n".join((intr, ques_cont[0], "\n".join([f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(options)]), instr))
        return sys_prompt + base_prompt, (first_lab, last_lab) if shuffle_order else (option_labels[0], option_labels[-1]) #, lab_A


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


    def _concat_token_blocks(self, first: dict, second: dict) -> dict:
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
    def format_chat(self, system_inputs: dict, tokenizer, **kwargs) -> dict:
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
        inputs = self._concat_token_blocks(system_inputs, user_inputs)
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
            self,
            model,
            tokenizer,
            inputs: dict,
            temperature: float = 0.1,
            max_len: int = None,
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
            max_len = int(input_len+input_len * 0.2) if not max_len else max_len
            with autocast("cuda"):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_len,
                    temperature=temperature
                )
            output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        torch.cuda.empty_cache()
        return output