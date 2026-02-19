import argparse
from pathlib import Path

from utils.cleaning import process_split
from utils.model import load_model_tokenizer, prepare_prompt

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


def run(splits: list[str], source: str, task: str):
    """
    Clean the MedMCQA dataset by processing each split (train, dev, test) and saving the cleaned version.
    :param splits: List of data splits to process.
    :param source: The source directory containing the original data files.
    :param task: The specific cleaning task to perform.
    :return:
    """
    Path.mkdir(Path("data/MedMCQA/cleaned"), exist_ok=True)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct")

    prompt = prepare_prompt(task, tokenizer)

    for split in splits:
        print(f"Processing split {split}")
        process_split(data_path, source, split, prompt, task, model, tokenizer)


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
        "--task",
        type=str,
        required=True,
        action=CheckTaskAction,
        choices=["question", "explanation", "classification"],
        help="The cleaning task to perform",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Running the data cleaning scrip with splits:", args.splits, "and task:", args.task)
    run(splits=args.splits, source=args.source, task=args.task)