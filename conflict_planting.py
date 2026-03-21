import argparse
import json
import os
from pathlib import Path
import re

from utils.model import _concat_token_blocks, generate_text, load_model_tokenizer

data_path = "data/MedMCQA"


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
        for split in splits:
            if "cleaned" in value:
                path = Path(data_path) / value / f"{split}_exp_unfinished_q_leaked.jsonl"
            else:
                path = Path(data_path) / value / f"{split}.jsonl"
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
    answ_options = re.split("(?i)[1-4a-d][).]", question)[1:]
    return [] if not answ_options else [opt.strip() for opt in answ_options if opt.strip()]


def get_answer_options(question_data: dict) -> list[str]:
    """
    Extract the answer options from a question data dict.
    :param question_data: A dict containing the question data, expected to have keys like "option1", "option2", etc.
    :return: A list of answer options extracted from the question data.
    """
    options = {"cop": [], "rem_ans_opts": []}

    cop = question_data.get("cop", "")-1 # Adjust cop index

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
    system_inputs = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return system_inputs

def prepare_user_inputs(_user_prompt, tokenizer, context: str, question: str, answer_options: dict) -> dict:

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
    parser = argparse.ArgumentParser(description="Clean MedMCQA dataset")
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
        "--source-type",
        choices=["original", "cleaned"],
        default="cleaned",
        help="Type of source data to process (default: cleaned)",
    )

    return parser.parse_args()


def save_generated_data(split: str, generated_data: list[dict]):
    output_file = Path(data_path) / "synthetic" / f"{split}_conflict_planted.jsonl"
    with open(output_file, "w") as f:
        for item in generated_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved generated data to {output_file}")

def retrieve_answer_from_raw(cop: int, answer_options: dict, mod_answer: str) -> str:
    """
    Retrieve the according answer option from the question data from the model raw answer.
    :param cop: The index of the correct answer option (0-based index)
    :param mod_answer: The raw answer string generated by the model
    :param answer_options: The dict of answer options extracted from the question data, containing the correct answer option and the remaining answer options.
    :return: The corresponding answer option value (e.g., "opa", "opb", etc.) that matches the model's answer.
    """
    for ans_type, answers in answer_options.items():
        if ans_type == "rem_ans_opts":
            for i, ans in enumerate(answers):
                if ans.lower() in mod_answer.lower():
                    return i+1 if cop <= i else i # If the model's answer matches one of the remaining answer options, return the corresponding index (adjusted for the correct answer index)
            if len(mod_answer) == 1 and mod_answer.lower() in ['a', 'b', 'c', 'd']:
                ans_index = ord(mod_answer.lower()) - ord('a')
                if ans_index < cop:
                    return ans_index
                elif ans_index >= cop:
                    return ans_index + 1
        elif ans_type == "cop":
            if answers.lower() in mod_answer.lower():
                return cop # If the model's answer matches the correct answer option, return the correct answer index
    return None # If no match is found, return None

def parse_output(output: str) -> dict:
    """
    Parse the generated output from the model to extract the conflict-planted context and the corresponding question-answer pair.
    :param output: The raw output string generated by the model.
    :return: A dict containing the parsed context, question, and answer.
    """
    mod_context = re.search(r"(?i)Context(.*):\s*(.*?)\s*(?:Question:|Correct Answer:)", output, re.DOTALL)
    mod_answer = re.search(r"(?i)\bCorrect Answer(.*):[\s\*]*(.*)", output)
    exp_note = re.search(r"(?i)Note:\s*(.*?)$", output, re.DOTALL)

    return {
        "mod_context": mod_context.group(2).strip() if mod_context else None,
        "mod_answer": mod_answer.group(2).strip() if mod_answer else None,
        "exp_note": exp_note.group(1).strip() if exp_note else None
    }


if __name__ == "__main__":
    args = parse_args()
    print("Running the conflict planting script:", args.splits, "and source:", args.source)
   
    Path.mkdir(Path("data/MedMCQA/synthetic"), exist_ok=True)
   
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct")
    print(model.config.name_or_path, "loaded successfully.")
   
    with open("utils/prompts/conflict_planting.txt", "r") as f:
        base_prompt = f.read()
    sys_prompt = prepare_system_inputs(tokenizer, re.split(r"\n(?=Context)", base_prompt)[0])
    _user_prompt = re.split(r"\n(?=Context)", base_prompt)[1]

    # Counters for statistics
    op_list_q_cnt = 0
    op_list_q_upd_cnt = 0
    null_upd_answer_cnt = 0
    null_upd_context_cnt = 0
    invalid_upd_answer_cnt = 0
    same_upd_answer_cnt = 0

    for split in args.splits:
        print(f"Processing {args.source_type} split {split} of {args.source} data...")
        data_file = Path(data_path) / args.source / f"{split}_exp_unfinished_q_leaked.jsonl"
        generated_data = []
        with open(data_file, "r") as f:
            count = 0
            for i, line in enumerate(f): # Process only the first 100 lines for testing
                if count >= 50:
                    break
                question_data = json.loads(line)
                exp_to_edit = question_data.get("exp_to_edit", "")
                if not exp_to_edit:
                    continue
                id = question_data.get("id", "")
                op_list_in_question = question_data.get("op_list_in_question", False)
                if op_list_in_question:
                    op_list_q_cnt +=1
                    if question_data.get("op_list_in_question_upd", False):
                        # If the updated question still contains the answer options list, take it. Otherwise, take the original question.
                        op_list_q_upd_cnt += 1
                        question_upd = question_data.get("question_upd", "")
                else: 
                    question_upd = question_data.get("question", "")
                exp_upd = question_data.get("exp_upd", "")
                answer_options = get_answer_options(question_data)
                cop = question_data.get("cop", "")-1 # Adjust cop index to match the answer options list index (0-based)
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
                if parsed_data["mod_context"] is None:
                    null_upd_context_cnt +=1
                    print(f"Could not parse a valid context from the model's output for question id {id}. Skipping this instance.")
                    continue

                if parsed_data["mod_answer"] is None:
                    null_upd_answer_cnt +=1
                    print(f"Could not parse a valid answer from the model's output for question id {id}. Skipping this instance.")
                    continue

                cop_upd = retrieve_answer_from_raw(cop, answer_options, parsed_data["mod_answer"])     
                if cop_upd is None:
                    invalid_upd_answer_cnt +=1
                    print(f"Could not retrieve a valid answer option from the model's output for question id {id}. Saving accordingly.")
                    # continue
                elif cop_upd == cop:
                    same_upd_answer_cnt +=1
                    print(f"The model's answer for question id {id} did not change the correct answer option. Skipping this instance.")
                    continue

                generated_data.append({
                    **question_data,
                     "cop_upd": cop_upd+1 if cop_upd is not None else None, # Adjust back to 1-based index for the updated correct answer
                       **parsed_data
                       })
                count += 1
        # Save statistics        
        print(f"Statistics for split {split}:")
        print(f"Total questions processed: {i+1}")
        print(f"Total questions with answer options list in question: {op_list_q_cnt}")
        print(f"Total questions with updated question still containing answer options list: {op_list_q_upd_cnt}")
        print(f"Total questions with null updated context parsed from model output: {null_upd_context_cnt}")
        print(f"Total questions with null updated answer parsed from model output: {null_upd_answer_cnt}")
        print(f"Total questions with invalid updated answer parsed from model output: {invalid_upd_answer_cnt}")
        print(f"Total questions with same updated answer as original: {same_upd_answer_cnt}")

        save_generated_data(split, generated_data)
    