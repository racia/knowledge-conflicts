import argparse
from pathlib import Path

from utils.cleaning import process_split
from utils.model import load_model_tokenizer, prepare_prompt

DATA_PATH = "data/MedMCQA"

def validate_args(args):
    allowed_splits = {"dev", "test", "train"}
    # values will be a list when nargs='+' is used
    items = args.splits if isinstance(args.splits, list) else [args.splits]
    for v in items:
        if v not in allowed_splits:
            raise SystemExit(f"invalid split: {v!r}")

    for split in args.splits:
        if "cleaned" in args.source:
            path = Path(DATA_PATH) / args.source / f"{split}_classification.jsonl"
        else:
            path = Path(DATA_PATH) / args.source / f"{split}.jsonl"

        if not path.exists():
            raise SystemExit(f"error: {path} does not exist")

    allowed_tasks = {"question", "explanation", "classification"}
    if args.task not in allowed_tasks:
        raise SystemExit(f"Unknown cleaning task: {args.task}, choose from {allowed_tasks}")

    if type(args.reverse) is not bool:
        raise SystemExit(f"Invalid value for --reverse: {args.reverse}, expected a boolean (True/False)")

    if args.filtering_ids and args.filtering_ids.endswith(".txt"):
        raise SystemExit(f"Invalid filtering IDs file name: {args.filtering_ids}. "
                         f"Please don't include the '.txt' extension, just provide the base name (e.g., 'empty_ids').")


def run(splits: list[str], source: str, task: str, reverse: bool = False, filtering_ids: str = None):
    """
    Clean the MedMCQA dataset by processing each split (train, dev, test) and saving the cleaned version.
    :param splits: List of data splits to process.
    :param source: The source directory containing the original data files.
    :param task: The specific cleaning task to perform.
    :param reverse: Whether to process the data in reverse order (default: False).
    :param filtering_ids: Optional name of the file containing entry IDs to filter and process (e.g., "empty_ids").
    :return: None
    """
    Path.mkdir(Path("data/MedMCQA/cleaned"), exist_ok=True)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct")

    prompt = prepare_prompt(task, tokenizer, filtering_ids)

    for split in splits:
        print(f"Processing split {split}")
        process_split(DATA_PATH, source, split, prompt, task, model, tokenizer,
                      reverse=reverse, filtering_ids=filtering_ids)


def parse_args():
    parser = argparse.ArgumentParser(description="Clean MedMCQA dataset")
    parser.add_argument(
        "--source",
        dest='source',
        type=str,
        required=True,
        help="Source directory containing the data files",
    )
    parser.add_argument(
        "--splits",
        dest='splits',
        nargs="+",
        default=["train", "dev", "test"],
        help="Data splits to process (default: all splits)",
    )
    parser.add_argument(
        "--task",
        dest='task',
        type=str,
        required=True,
        choices=["question", "explanation", "classification"],
        help="The cleaning task to perform",
    )
    parser.add_argument(
        "--reverse",
        dest='reverse',
        action='store_true',
        help="Whether to process the data in reverse order (default: False)",
        default=False,
    )
    parser.add_argument(
        "--filtering_ids",
        dest='filtering_ids',
        type=str,
        required=False,
        help="File name containing entry IDs to filter and process (e.g., 'empty_ids')",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    print(f"Running the data cleaning scrip with splits: {args.splits}, task: '{args.task}', reverse: {args.reverse}, filtering_ids: {args.filtering_ids}")
    run(splits=args.splits, source=args.source, task=args.task, filtering_ids=args.filtering_ids, reverse=args.reverse)