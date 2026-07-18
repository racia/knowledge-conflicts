import argparse
import json
import os
from pathlib import Path
import re
from omegaconf import DictConfig, OmegaConf

from utils.model import _concat_token_blocks, generate_text, load_model_tokenizer


class CheckSplitsAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        allowed = {"dev", "test", "train"}
        # values will be a list when nargs='+' is used
        items = values if isinstance(values, list) else [values]
        for v in items:
            if v not in allowed:
                raise argparse.ArgumentError(self, f"invalid split: {v!r}")
        setattr(namespace, self.dest, values)


class CheckSourceAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        splits = getattr(namespace, "splits", [])
        source = getattr(namespace, "source", "") # This retrieves the default value of the source argument
        cfg = getattr(namespace, "cfg", None)
        for split in splits:
            if "cleaned" in value:
                path = Path(cfg.data.path) / value / f"{split}.jsonl" # Old: _exp_unfinished_q_leaked.jsonl
            else:
                path = Path(cfg.data.path) / value / f"{split}.jsonl"
            if not path.exists():
                raise argparse.ArgumentError(self, f"{path} does not exist")
        setattr(namespace, self.dest, value)


class CheckTaskAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        allowed = {"question", "explanation", "classification"}
        if value not in allowed:
            raise ValueError(f"Unknown cleaning task: {value}, choose from {allowed}")
        setattr(namespace, self.dest, value)


def parse_question(question: str) -> list[str]:
    """
    Parse the question string to extract answer options if they are contained within the question text.
    This function looks for patterns like "1) (or 1.) option A", "a) (or a.) option A",
    :param question: The question string to be parsed.
    :return: A list of answer options extracted from the question string, or an empty list if no options are found.
    """
    answ_options = re.split("(?i)[1-4a-d][).]", question)[1:]
    return [] if not answ_options else [opt.strip() for opt in answ_options if opt.strip()]


def get_answer_options(cop: int, question_data: dict) -> list[str]:
    """
    Extract the correct and remaining answer options from a question data dict.
    :param cop: The index of the correct answer option. (0-indexed)
    :param question_data: A dict containing the question data, expected to have keys like "op1", "op2", etc.
    :return: A dict containing the correct and remaining answer options based on the provided question data. 
    """
    options = {"cop": [], "rem_ans_opts": []}

    answ_opts = parse_question(question_data.get("question", ""))
    
    # if answ_opts: # If answers contained, use them as answer options, otherwise use the opa, opb values as answer options
    #     options["cop"] = answ_opts[cop] # CAUTION: this does not actually depict the provided answer options, rather the contained choices in the question. 
    #     options["rem_ans_opts"] = [opt for i, opt in enumerate(answ_opts) if i != cop] # The remaining answer options are those that are not the correct one
    # else:
    # Try without parsing the question, directly use the "op1", "op2", etc. values as answer options
    options["cop"] = question_data.get("op{}".format(chr(ord('a') + cop)), "") # The correct answer option is determined by the "cop" index, which corresponds to "opa", "opb", etc.
    options["rem_ans_opts"] = [question_data.get("op{}".format(chr(ord('a') + i)), "") for i in range(4) if i != cop] # The remaining answer options are those that are not the correct one, determined by the "cop" index
    return options


def prepare_system_inputs(tokenizer, system_prompt: str) -> dict:
    """
    :param tokenizer: The tokenizer to use for processing the system prompt.
    :param system_prompt: The raw system prompt string to be processed and tokenized.
    :return: A dict containing the tokenized system prompt inputs, ready to be fed into the model.
    """
    system_inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return system_inputs


def prepare_user_inputs(_user_prompt, tokenizer, context: str, question: str, answer_options: dict) -> dict:
    """
    :param _user_prompt: The raw user prompt template containing placeholders for context, question, and answer options.
    :param tokenizer: The tokenizer to use for processing the user prompt.
    :param context: The original context from the question data to be inserted into the user prompt
    :param question: The original question from the question data to be inserted into the user prompt
    :param answer_options: A dict containing the correct answer and remaining answer options to be inserted
    :return: A dict containing the tokenized user prompt inputs, ready to be fed into the model.
    """
    prompt = _user_prompt.replace("<CONTEXT>", context).replace("<QUESTION>", question)
    prompt = prompt.replace("<ANSWER_OPTIONS>", "{}, {} or {}".format(answer_options["rem_ans_opts"][0], answer_options["rem_ans_opts"][1], answer_options["rem_ans_opts"][2]))
    
    user_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return user_inputs


def parse_args():
    parser = argparse.ArgumentParser(description="Plant conflicts in MedMCQA dataset")
    parser.add_argument(
        "--config",
        help="Configuration file for the conflict planting process"
    )
    
    # Parse config first to get cfg object
    args, remaining = parser.parse_known_args()
    cfg = OmegaConf.load(args.config) if args.config else None

    # Create a new parser with cfg available to custom actions
    parser = argparse.ArgumentParser(description="Plant conflicts in MedMCQA dataset")
    parser.set_defaults(cfg=cfg)
    
    parser.add_argument(
        "--splits",
        nargs="+",
        action=CheckSplitsAction,
        default=["train", "dev", "test"],
        help="Data splits to process (default: all splits)",
    )
    parser.add_argument(
        "--source",
        action=CheckSourceAction,
        default="original/clean_spaces_id",
        help="Source directory containing the data files",
    )
    parser.add_argument(
        "--config",
        help="Configuration file for the conflict planting process"
    )
    
    return parser.parse_args()


def save_generated_data(split: str, data_dict: dict, cfg: DictConfig):
    """
    Save the generated data with conflict-planted contexts and corresponding question-answer pairs to a JSONL file.
    :param split: The data split (e.g., "train", "dev", "test")
    :param data_dict: A dict containing the generated data to be saved.
    :param cfg: The configuration object containing the data path for saving the generated data.
    """
    output_file = Path(cfg.data.path) / "synthetic" / f"{split}_conflict_planted_{cfg.data.part_idx}.jsonl"
    with open(output_file, "a") as f:
        f.write(json.dumps(data_dict) + "\n")
    #print(f"Saved generated data to {output_file}")


def retrieve_answer_from_raw(cop: int, mod_answer: str, answer_options: dict) -> str:

    """
    Retrieve the corresponding answer option (e.g., "opa", "opb", etc.) based on the generated answer text.
    :param mod_answer: The generated answer text from the model.
    :param answer_options: A dict containing the correct answer and remaining answer options.
    :return: The corresponding answer option key (e.g., "opa") if found, otherwise None.
    """
    if mod_answer is None:
        return None
    elif answer_options["cop"].lower() in mod_answer.lower():
        return cop
    for i, ans in enumerate(answer_options["rem_ans_opts"]):
        # print(f"Checking if option '{ans}' matches the generated answer '{mod_answer}'...")
        if ans.lower() in mod_answer.lower():
            return i+1 if cop <= i else i # Adjust index to account for the position of the correct answer in the options list
        elif len(mod_answer) == 1 and mod_answer.lower() in ['a', 'b', 'c', 'd']:
            ans_index = ord(mod_answer.lower()) - ord('a')
            return ans_index if ans_index <= cop else ans_index + 1
    return None


def parse_output(output: str) -> dict:
    """
    Parse the generated output from the model to extract the conflict-planted context and the corresponding question-answer pair.
    :param output: The raw output string generated by the model.
    :return: A dict containing the parsed context, question, and answer.
    """
    mod_context = re.search(r"(?i)(Context(.*?):\s*)+([\s\S]*?)(?:\n\s*\n)", output)
    mod_answer = re.search(r"(?i)\bCorrect Answer(.*):[\s\*]*(.*)", output)
    exp_note = re.search(r"(?i)Note:\s*(.*?)$", output, re.DOTALL)

    return {
        "mod_context": mod_context.group(3).strip() if mod_context else None,
        "mod_answer": mod_answer.group(2).strip() if mod_answer else None,
        "exp_note": exp_note.group(1).strip() if exp_note else None
    }

def partition_data(data_file: Path, num_parts: int, part_idx: int) -> list[str]:
    """
    Partition the data file into specified number of parts and return the portion corresponding to the given part index.
    :param data_file: The path to the data file to be partitioned.
    :param num_parts: The total number of parts to divide the data into.
    :param part_idx: The index of the part to be returned (1-indexed).
    :return: A list of lines corresponding to the specified part index.
    """
    with open(data_file, "r") as f:
        data_lines = f.readlines()

    if num_parts <= 0:
        return data_lines
    part_size = len(data_lines) // num_parts
    start_idx = part_size * (part_idx - 1)
    end_idx = start_idx + part_size if part_idx < num_parts else len(data_lines)
    return data_lines[start_idx:end_idx]

def check_existing_generated_data(split: str, cfg: DictConfig, data_part: list) -> int:
    """
    Check for existing generated data for the given split and return the count of already processed instances.
    :param split: The data split (e.g., "train", "dev", "test") to check for existing generated data.
    :param cfg: The configuration object containing the data path for checking existing generated data.
    :param data_part: The portion of the data file being processed.
    :return: The count of already processed instances based on existing generated data files.
    """
    output_file = Path(cfg.data.path) / "synthetic" / f"{split}_conflict_planted_{cfg.data.part_idx}.jsonl"
    if output_file.exists():
        with open(output_file, "r") as json_f:
            existing_data = [json.loads(line) for line in json_f]
        last_processed_id = existing_data[-1].get("id", "N/A") if existing_data else "N/A"
        processed_count = data_part.index(next((line for line in data_part if json.loads(line).get("id", "") == last_processed_id), None)) + 1 if last_processed_id != "N/A" else 0
        print(f"Existing generated data found for split {split}. Last processed question ID: {last_processed_id}. Resuming from the next instance.")
        return processed_count
    return 0


if __name__ == "__main__":
    args = parse_args()
    conf = OmegaConf.load(args.config)

    Path.mkdir(Path(f"{conf.data.path}/synthetic"), exist_ok=True)
   
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_tokenizer(conf.model.name)
    print(model.config.name_or_path, "loaded successfully.")
   
    with open(conf.task.prompt, "r") as f:
        base_prompt = f.read()
    sys_prompt = prepare_system_inputs(tokenizer, re.split(r"\n(?=Context)", base_prompt)[0])
    _user_prompt = re.split(r"\n(?=Context)", base_prompt)[1]

    # Counters for statistics
    no_exp_count = 0
    no_exp_upd_count = 0
    op_list_q_cnt = 0
    op_list_q_upd_cnt = 0
    null_upd_answer_cnt = 0
    null_upd_context_cnt = 0
    invalid_upd_answer_cnt = 0
    same_upd_answer_cnt = 0

    for split in args.splits:
        print(f"Processing {split} split of {args.source} data from {conf.data.path}...")
        data_file = Path(conf.data.path) / args.source / f"{split}_exp_que_upd.jsonl"
        data_part = partition_data(data_file, num_parts=conf.data.num_parts, part_idx=conf.data.part_idx)
        processed_count = check_existing_generated_data(split, conf, data_part)
        data_portion = data_part[processed_count:conf.data.num_samples] if conf.data.num_samples > 0 else data_part[processed_count:]
        print(f"Processing {len(data_portion)} of {len(data_part)} instances in the partitioned data.")
        
        generated_data = []
        count = 0
        for i, line in enumerate(data_portion):
            question_data = json.loads(line)
            id = question_data.get("id", "")
            exp_to_edit = question_data.get("exp_to_edit", "")
            if not exp_to_edit:
                no_exp_count += 1
                print(f"No explanation provided for question id {id}. Skipping this instance.")
                continue
            exp_upd = question_data.get("exp_upd", "")
            if not exp_upd:
                no_exp_upd_count += 1
                print(f"No explanation update provided for question id {id}. Skipping this instance.")
                continue
            cop = question_data.get("cop", "")-1 # Adjust cop index to match the answer options list index (0-based)
            answer_options = get_answer_options(cop, question_data)
            question_upd = question_data.get("question", "")
            
            op_list_in_question = question_data.get("op_list_in_question", False)
            if op_list_in_question:
                op_list_q_cnt +=1
                if question_data.get("op_list_in_question_upd", False):
                    # If the updated question still contains the answer options list, take it. Otherwise, leave as original question.
                    op_list_q_upd_cnt += 1
                    question_upd = question_data.get("question_upd", "")
            
            user_inputs = prepare_user_inputs(
                _user_prompt, 
                tokenizer, 
                exp_upd, 
                question_upd,
                answer_options
                )
            inputs = _concat_token_blocks(sys_prompt, user_inputs)
            output = generate_text(model, tokenizer, inputs)
            print("Generated output:\n", output)
            parsed_data = parse_output(output)
            print("Parsed data:\n", parsed_data)
            cop_upd = retrieve_answer_from_raw(cop, parsed_data["mod_answer"], answer_options)     
            if parsed_data["mod_context"] is None:
                null_upd_context_cnt +=1
                print(f"Could not parse a modified context from the output for question ID {id}. Skipping this instance.")
                continue
            if parsed_data["mod_answer"] is None:
                null_upd_answer_cnt +=1
                print(f"Could not parse a modified answer from the output for question ID {id}. Skipping this instance.")
                continue
            elif cop_upd == cop:
                same_upd_answer_cnt +=1
                print(f"The modified answer did not change the correct answer option for question ID {id}. Skipping this instance.")
                continue
            elif cop_upd is None:
                invalid_upd_answer_cnt +=1
                print(f"Could not retrieve a valid answer option from the modified answer for question ID {id}. Saving accordingly.")
                
            generated_data.append({
                **question_data,
                **parsed_data,
                "cop_upd": cop_upd+1 if cop_upd is not None else None, # Adjust back to 1-based index for output
                })
            count+=1
            # Save directly after processing each question to avoid data loss in case of interruptions, and to monitor progress on larger datasets
            save_generated_data(split, {**question_data, **parsed_data, "cop_upd": cop_upd+1 if cop_upd is not None else None}, conf)

        # Save statistics        
        print(f"Statistics for split {split}:")
        print(f"Total questions processed: {i+1}")
        print(f"Total questions with no explanation provided: {no_exp_count}")
        print(f"Total questions with no explanation update provided: {no_exp_upd_count}")
        print(f"Total questions with answer options list in question: {op_list_q_cnt}")
        print(f"Total questions with updated question still containing answer options list: {op_list_q_upd_cnt}")
        print(f"Total questions with null updated context parsed from model output: {null_upd_context_cnt}")
        print(f"Total questions with null updated answer parsed from model output: {null_upd_answer_cnt}")
        print(f"Total questions with invalid updated answer parsed from model output: {invalid_upd_answer_cnt}")
        print(f"Total questions with same updated answer as original: {same_upd_answer_cnt}")

        #save_generated_data(split, generated_data, conf)
        print(f"Processed and saved {len(generated_data)} total questions with updated and (valid) answers.")    