import json
import numpy as np
import torch
import re
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

import tracemalloc
from utils.model import load_model_tokenizer, prepare_prompt#, extract_choice, run_model
import argparse
from omegaconf import DictConfig, OmegaConf


def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config


def process_samples(samples):
    print(f"Processing {len(samples)} samples...")
    processed = []
    for sample in samples:
        processed.append({
            "id": sample["id"],
            "question": sample["question_upd"],
            "opa": sample["opa"],
            "opb": sample["opb"],
            "opc": sample["opc"],
            "opd": sample["opd"],
            "cop": sample["cop"]
        })
    return processed

def build_prompt(sys_prompt, sample, cfg: DictConfig):
    if cfg.mcq.exp:
        exp_str = sample.get("exp", "")
        if exp_str:
            return prepare_prompt(cfg.task.type, cfg.prompt_path, tokenizer=tokenizer, sys_prompt=sys_prompt, sample_data=sample, exp=exp_str)
    return prepare_prompt(cfg.task.type, cfg.prompt_path, tokenizer=tokenizer, sys_prompt=sys_prompt, sample_data=sample)


def extract_choice(text):
    match = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else None

def run_model(prompt):
    with torch.no_grad():
        sys_prompt, prompt_body = prompt.split("\n\n", 1)

        prompt_inputs = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_prompt, "name": "system"}, 
             {"role": "user", "content": prompt_body, "name": "user"}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_str = tokenizer.decode(prompt_inputs["input_ids"][0], skip_special_tokens=True)
        inputs = prompt_inputs.to(model.device)
        with torch.autocast("cuda"):
            outputs = model.generate(**inputs, max_new_tokens=10)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded[len(prompt_str):].strip()

def evaluate(samples, cfg: DictConfig):
    gold, preds, outputs = [], [], []
    sys_prompt = "You are a helpful and precise medical assistant for answering multiple-choice questions. Always think step by step."
    for sample in samples:
        prompt = build_prompt(sys_prompt, sample, cfg=cfg)
        sys_prompt = None
        print("Running model with prompt:")
        print(prompt)
        model_output = run_model(prompt)
        pred_choice = extract_choice(model_output)
        try:
            assert sample["cop"] in [1, 2, 3, 4], f"Invalid cop value: {sample['cop']}"
            assert pred_choice in ["A", "B", "C", "D"], f"Model output does not contain a valid choice: {model_output}"
        except AssertionError as e:
            # Convert to nan value for accuracy calculation, but still include in outputs for analysis
            pred_choice = None
        
        gold_choice = chr(ord("A") + sample["cop"] - 1)
        
        gold.append(gold_choice)
        preds.append(pred_choice if pred_choice is not None else np.nan)

        outputs.append({
            "id": sample["id"],
            "model_output": model_output,
            "pred_choice": pred_choice,
            "gold_choice": gold_choice
        })

    accuracy = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")
    return outputs, accuracy, f1

if __name__ == "__main__":
    # starting the monitoring
    tracemalloc.start()
    # Parse args
    parser = argparse.ArgumentParser(description="Probe MCQA for conflicting data.")
    parser.add_argument("--config",
                        help="Configuration file for the MCQA probing")
    args = parser.parse_args()

    config = load_config(args.config)
    # Model configuration
    model_name = config.model.name
    model, tokenizer = load_model_tokenizer(model_name)
    # Data configuration
    data_path = config.data.path
    file_name = config.data.file_name
    data_file = Path(data_path) / file_name
    # Task configuration
    task_type = config.task.type
    task_name = config.task.name
    task_prompt = config.task.prompt
    # Task-specific configuration
    include_exp = config.mcq.context

    with open(data_file, "r") as f:
        samples = [json.loads(line) for line in f]

    num_samples = config.data.num_samples
    if num_samples != -1:
        samples = samples[:num_samples]

    # Empty torch cache to get accurate memory usage
    print("Emptying torch cache...")
    torch.cuda.empty_cache()

    samples = process_samples(samples)
    outputs, accuracy, f1 = evaluate(samples, cfg=config)
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    with open(f"outputs{'_exp' if include_exp else ''}.json", "w") as f:
        json.dump(outputs, f, indent=2)

    # displaying the memory usage
    print(tracemalloc.get_traced_memory())

    # stopping the library
    tracemalloc.stop()

