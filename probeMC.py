from collections import defaultdict
import json
import random
import time
import datetime as dt
import numpy as np
import torch
import re
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import tracemalloc
import sys

from transformers import pipeline
from utils.cleaning import DataCleaner
from utils.model import ModelLoader
import argparse
from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config
 
def process_samples(samples, shuffle_cop: bool = True):
    print(f"Processing {len(samples)} samples...")
    processed = []
           
    for sample in samples:
        org_cop = chr(ord("a")+int(sample["cop"])-1)
        org_cop_ans = sample["op{}".format(org_cop)]
        # Draw random samples from ans options excluding cop
        rand_op_idx = random.choices(["a", "b", "c", "d"], weights=[i!=sample["cop"]-1 for i in range(0,4)], k=1)[0]
        new_cop = 1 + ord(rand_op_idx) - ord("a")
        rem_op_ans = [sample["op{}".format(chr(ord("a")+i))] if i != new_cop-1 else org_cop_ans for i in range(0,4)]
        
        processed.append({
            "id": sample["id"],
            "question": sample["question_upd"],
            "exp": sample["exp"], # TODO: Applicability to exp(_upd)
            "opa": sample[f"op{rand_op_idx}"] if "a" == org_cop else rem_op_ans[0],
            "opb": sample[f"op{rand_op_idx}"] if "b" == org_cop else rem_op_ans[1],
            "opc": sample[f"op{rand_op_idx}"] if "c" == org_cop else rem_op_ans[2],
            "opd": sample[f"op{rand_op_idx}"] if "d" == org_cop else rem_op_ans[3],
            "cop_new": new_cop
        } if shuffle_cop else {
            "id": sample["id"],
            "question": sample["question_upd"],
            "exp": sample["exp"],
            "opa": sample["opa"],
            "opb": sample["opb"],
            "opc": sample["opc"],
            "opd": sample["opd"],
            "cop_new": sample["cop"]
        })

        try:
            assert sample[f"op{rand_op_idx}"] != rem_op_ans[new_cop-1]
        except:
            print(f"Same original and new cop detected for: ",sample[f"op{rand_op_idx}"], rem_op_ans[new_cop-1])
        # swp_op_ans if org_cop_ans == sample["opa"] else sample_rest[0]
        # sample_ints = [0,1,3] if not rands_idx-1, random.shuffle(sample_ints)
        # [sample[f"op{}"] for i in range(0,4) if i+1 != rand]
        # Sample from rest of ans without swp_ans_op: [f"op{chr(ord("a"))+next(sample_ints)}"]
    return processed

def build_prompt(task: str, prompt: str, sys_prompt, sample, shuffle_order: bool = False):
    if cfg.mcq.exp:
        exp_str = sample.get("exp", "") #TODO: for compatibility for exp and exp_upd
        if exp_str:
            return model_loader.prepare_prompt(task, prompt, sys_prompt=sys_prompt, processed=sample, exp_str=exp_str, shuffle_order=shuffle_order)
    return model_loader.prepare_prompt(task, prompt, sys_prompt=sys_prompt, processed=sample, shuffle_order=shuffle_order)


def extract_choice(text):
    match = re.search(r"\b([ABCD])\b", text.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else None

def run_model(prompt, model=None, tokenizer=None, pipeline=None):
    with torch.no_grad():
        full_prompt = prompt.split("\\n\\n", 1)
        # print(f"Full prompt, ", full_prompt)
        if not pipeline:
            model_inputs = tokenizer.apply_chat_template(
                [{"role": "system", "content": full_prompt[0], "name": "system"},
                {"role": "user", "content": full_prompt[1], "name": "user"}],
                add_generation_prompt=cfg.model.add_generation_prompt, # TODO: What with True for start generation?
                tokenize=cfg.model.tokenize,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

        elif pipeline:
            model_inputs = pipeline.tokenizer.apply_chat_template(
                [{"role": "system", "content": full_prompt[0]},
                 {"role": "user", "content": full_prompt[1]}],
            tokenize = False,
            add_generation_prompt=True,
            # return_tensors="pt"
            )
            terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            model_inputs = tokenizer([model_inputs], return_tensors="pt").to(model.device)
        
        prompt_str = tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True)
        input_length = model_inputs["input_ids"].shape[1]#, skip_special_tokens=True)
        try:
            assert len(prompt_str) == input_length
        except:
            # print(f"Length mismatch of Prompt str: {len(prompt_str), prompt_str[:5], prompt_str[-5:]} with input length: {input_length}") # Input length around 4* smaller
            pass
        with torch.autocast("cuda"):
            if pipeline:
                outputs = pipeline(
                    prompt, # Chat says to use input_ids, but pipeline expects str input
                    max_new_tokens=256,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.0,
                    top_p=0.9,
                )                
            else:
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=10,
                    temperature=0.7,
                    do_sample=False, # For deterministic outputs
                )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True) 
        print("original model output: ", decoded)
        return decoded[len(prompt_str):].strip()

def evaluate(task: str, prompt: str, samples, model=None, tokenizer=None, pipeline=None, shuffle_order: bool = False):
    gold, preds, outputs = [], [], []
    gold_count, pred_count, labels_stats = defaultdict(int), defaultdict(int), defaultdict(lambda: defaultdict(int))
    none_counter = 0
    sys_prompt = "You are a helpful and precise medical assistant for answering multiple-choice questions. Always think step by step."
    for i, sample in enumerate(samples):
        prompt, (first_lab, last_lab) = build_prompt(task_type, task_prompt, sys_prompt, sample=sample, shuffle_order=shuffle_order)
        labels_stats[first_lab]["first"] += 1
        labels_stats[last_lab]["last"] += 1
        sys_prompt = None
        print(f"Running model with prompt: {prompt}")
        model_output = run_model(prompt, model=model, tokenizer=tokenizer, pipeline=pipeline)
        pred_choice = extract_choice(model_output)
        gold_choice = chr(ord("A") + sample["cop_new"] - 1) 
        gold.append(gold_choice) 
        gold_count[gold_choice] += 1
        try:
            assert sample["cop_new"] in [1, 2, 3, 4], f"Invalid cop value: {sample['cop']}"
            assert pred_choice in ["A", "B", "C", "D"], f"Model output does not contain a valid choice: {model_output}"
            preds.append(pred_choice)
            pred_count[pred_choice] += 1
        except AssertionError as e:
            # Convert to nan value for accuracy calculation, but still include in outputs for analysis
            # pred_choice = None
            none_counter += 1
            pass
        print(none_counter, model_output)
        
        outputs.append({
            "id": sample["id"],
            "model_output": model_output,
            "pred_choice": pred_choice,
            "gold_choice": gold_choice,        
        })

    outputs_stats = {"pred_count": pred_count,
                    "gold_count": gold_count, 
                    "inv_outputs": none_counter,
                    }
    accuracy = accuracy_score(gold, preds)
    f1 = f1_score(gold, preds, average="macro")
    conf_matr = confusion_matrix(preds, gold, labels=["A", "B", "C", "D"])
    return outputs, accuracy, f1, conf_matr, outputs_stats, labels_stats

if __name__ == "__main__":
    # Log to console/file
    orig_stdout = sys.stdout
    
    # starting the time/mem monitoring
    tracemalloc.start()
    start_time = time.time()
    # Parse args
    parser = argparse.ArgumentParser(description="Probe MCQA for conflicting data.")
    parser.add_argument("--config",
                        help="Configuration file for the MCQA probing",
                        default="configs/probeMCQ-exp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    runs = cfg.run.num_runs
    shuffle_cop = cfg.data.shuffle_cop
    shuffle_order = cfg.data.shuffle_order

    today = str(dt.datetime.today()).split()
    outputs_path = Path(cfg.outputs.path, today[0], today[1].split(".")[0] if cfg.outputs.clock_time else "")
    Path.mkdir(outputs_path, parents=True, exist_ok=True)

    for run in range(runs):
        start_time_run = time.time()

        print(f"Starting run {run}/{runs}")
        run_path = Path(outputs_path, f"run{run}")
        Path.mkdir(run_path, parents=True, exist_ok=True)
        

        # Data configuration
        data_path = cfg.data.path
        file_name = cfg.data.file_name
        data_file = Path(data_path) / file_name
        
        # Task configuration
        task_type = cfg.task.type
        task_name = cfg.task.name
        task_prompt = cfg.task.prompt
        
        # Task-specific configuration
        include_exp = cfg.mcq.exp
        iterations = cfg.run.num_iterations
        
        # Instantiate cleaner for potential use in prompt construction
        data_cleaner = DataCleaner()

        model_names = cfg.model.name if isinstance(cfg.model.name, (list, ListConfig)) else [cfg.model.name]
        for model_name in model_names:

            cfg.model.name = model_name
            # Model configuration
            model_loader = ModelLoader(load_model=model_name, load_tokenizer=cfg.model.load_tokenizer, load_pipeline=cfg.model.load_pipeline)
            model, tokenizer, pipeline = model_loader.load_model_tokenizer(model_name=model_name)
            model_path = Path(run_path, model_name.split("/")[0].lower())
            Path.mkdir(model_path, parents=True, exist_ok=True)
            
            if cfg.outputs.print_to_file:
                outputs_file = open(Path(run_path, f"outputs_{run}.txt"), "w") # TODO: print to file for each model in run
                model_name_clean = model_name.replace("/", "_").replace(":", "_")
                model_file = open(Path(model_path, f"model_{model_name_clean}.txt"), "w")
                sys.stdout = outputs_file if not cfg.outputs.file_per_model else model_file
            else:
                sys.stdout = orig_stdout 

            print(f"Running model: {model_name} with config: {OmegaConf.to_container(cfg, resolve=True)}")

            model_loader.set_data_cleaner(data_cleaner)
            data_cleaner.set_model_loader(model_loader)

            with open(data_file, "r") as f:
                samples = [json.loads(line) for line in f]

            num_samples = cfg.data.num_samples
            if num_samples != -1:
                samples = samples[:num_samples]
            cfg.data.num_samples = len(samples) # Update config with actual number of samples loaded

            # Empty torch cache to get accurate memory usage
            print("Emptying torch cache...")
            torch.cuda.empty_cache()

            samples = process_samples(samples, shuffle_cop=shuffle_cop)
            for i in range(iterations):
                # Reset pruned_exp in model_loader's for correct monitoring
                data_cleaner.reset_pruned_exp()
                
                print(f"Running iteration {i+1}/{iterations}")
                print(f"Evaluating model with {tokenizer.__class__.__name__}, {model.__class__.__name__}, {pipeline.__class__.__name__ if pipeline else 'No Pipeline'}")
                outputs, accuracy, f1, conf_matr, outputs_stats, labels_stats = evaluate(task_type, task_prompt, samples, model=model, tokenizer=tokenizer, pipeline=pipeline, shuffle_order=shuffle_order)
                print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
                
                num_first = sum([labels_stats[opt]["first"] for opt in labels_stats]) #TODO: .values() would also work
                num_last = sum([labels_stats[opt]["last"] for opt in labels_stats])
                
                lab_pos_stats = {opt: {"first": "%.2f"%(labels_stats[opt]["first"]/num_first), "last": "%.2f"%(labels_stats[opt]["last"]/num_last)} for opt in labels_stats}
                print(f"Label position stats: {lab_pos_stats}")

                num_golds = sum(outputs_stats["gold_count"].values())
                num_preds = sum(outputs_stats["pred_count"].values())
                pred_stats = {opt: [{"pred_opt": "%.2f"%(outputs_stats["pred_count"][opt]/num_preds), "corr_opt": "%.2f"%(outputs_stats["gold_count"][opt]/num_golds)}] for opt in outputs_stats["gold_count"]}
                pred_stats["inv"] = [{"count": outputs_stats["inv_outputs"], "rate": "%.2f"%(outputs_stats["inv_outputs"] / num_golds)}] # Ratio of inv outputs of samples

                precision = np.array([conf_matr[i][i] / conf_matr.sum(axis=0)[i] for i in range(len(outputs_stats["pred_count"]))])
                recall = np.array([conf_matr[i][i] / conf_matr[i].sum() for i in range(len(outputs_stats["pred_count"]))])
                [pred_stats[opt].append({"pr": "%.2f"%(precision[i]), "rc": "%.2f"%(recall[i])}) for i, opt in enumerate(sorted(outputs_stats["pred_count"]))]
                print(f"Prediction stats: {pred_stats}")
                print(f"Confusion matrix:\n{conf_matr}")

                # Dump outputs, accuracy and f1 for analysis
                result_data = {
                    "time": str(dt.datetime.today()).split(".")[0],
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "outputs": outputs,
                    "accuracy": accuracy,
                    "f1": f1,
                    "pruned_exp": model_loader.data_cleaner.pruned_exp,
                    "pred_stats": pred_stats,
                    "lab_pos_stats": lab_pos_stats,
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                }
                with open(f"{model_path}/outputs{'_exp' if include_exp else ''}_{i+1}.json", "w") as f:
                    json.dump(result_data, f, indent=2)

        end_time_run = time.time() - start_time_run
        print(f"--- run {run} completed in {end_time_run:.2f} seconds ---")
        
    end_time = time.time() - start_time
    # displaying the memory usage
    print(f"Total memory usage: {tracemalloc.get_traced_memory()}")
    print(f"--- Total %s time ---" % (end_time))

    # stopping the library
    tracemalloc.stop()

    # Reset stdout
    sys.stdout = orig_stdout

