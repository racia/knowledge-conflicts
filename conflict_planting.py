import argparse
import json
import os
from pathlib import Path
import re

from google import genai


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
        source = getattr(namespace, "source", "")
        for split in splits:
            if "cleaned" in source:
                path = Path(data_path) / value / f"{split}_classification.jsonl"
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
    answ_options = re.split("[1234][).]", question)[1:]
    return [] if not answ_options else [opt.strip() for opt in answ_options if opt.strip()]


def get_answer_options(question_data: dict) -> list[str]:
    """
    Extract the answer options from a question data dict.
    :param question_data: A dict containing the question data, expected to have keys like "option1", "option2", etc.
    :return: A list of answer options extracted from the question data.
    """
    options = {"cop": [], "rem_ans_opts": []}

    cop = question_data.get("cop", "")

    answ_opts = parse_question(question_data.get("question", ""))
    
    if answ_opts: # If answers contained, use them as answer options, otherwise use the opa, opb values as answer options
        options["cop"] = answ_opts[cop] # The correct answer option is determined by the "cop" index
        options["rem_ans_opts"] = [opt for i, opt in enumerate(answ_opts) if i != cop] # The remaining answer options are those that are not the correct one
    else:
        options["cop"] = question_data.get("op{}".format(chr(ord('a') + cop)), "") # The correct answer option is determined by the "cop" index, which corresponds to "opa", "opb", etc.
        options["rem_ans_opts"] = [question_data.get("op{}".format(chr(ord('a') + i)), "") for i in range(4) if i != cop] # The remaining answer options are those that are not the correct one, determined by the "cop" index
    return options


def plant(context: str, question: str, answer_options: dict,):
    """
    Generate conflict-planted data by sythesizing context according to false answer options. Save generated data in the synthetic folder.
    :return:
    """
    Path.mkdir(Path("data/MedMCQA/synthetic"), exist_ok=True)
    
    print("Loading model and tokenizer...")
    client = genai.Client(api_key=os.environ.get("GENAI_API_KEY"))
    model = "gemini-2.5-flash-lite"
    # model, tokenizer = load_model_tokenizer("gemini-2.5-flash-lite")

    with open("utils/prompts/conflict_planting.txt", "r") as f:
        base_prompt = f.read()
    prompt = base_prompt.replace("<CONTEXT>", context).replace("<QUESTION>", question)
    prompt = prompt.replace("<ANSWER_OPTIONS>", "{}, {} or {}".format(answer_options["rem_ans_opts"][0], answer_options["rem_ans_opts"][1], answer_options["rem_ans_opts"][2]))
    print("Generated prompt for conflict planting:\n", prompt)
    response = client.models.generate_content(model=model, contents=prompt)
    print("Generated response:\n", response.text)

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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Running the conflict planting script:", args.splits, "and source:", args.source)
    for split in args.splits:
        print(f"Processing split {split}")
        data_file = Path(data_path) / args.source / f"{split}.jsonl"
        with open(data_file, "r") as f:
            count = 0
            for line in f: # Process only the first 100 lines for testing
                if count >= 100:
                    break
                question_data = json.loads(line)
                context = question_data.get("context", "")
                question = question_data.get("question", "")
                answer_options = get_answer_options(question_data)
                plant(context, question, answer_options)
    