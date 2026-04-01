import os
import sys
import logging
import transformers
import datasets
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from transformers import HfArgumentParser, TrainingArguments
from utils.arguments import ModelArguments, DataTrainingArguments


logger = logging.getLogger(__name__)


def parse_args(fn):
    def inner():
        # See all possible arguments in src/transformers/training_args.py
        # or by passing the --help flag to this script.
        # We now keep distinct sets of args, for a cleaner separation of concerns.

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        return fn(model_args, data_args, training_args)
    return inner


def setup_logging(training_args):
    """Set up logging"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")


def delete_prev_model(path: str):
    """Delete previously saved model"""
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)


def detect_last_checkpoint(training_args):
    """Detect the last saved checkpoint"""
    required_attrs = ("output_dir", "do_train", "overwrite_output_dir", "resume_from_checkpoint")
    assert all(hasattr(training_args, attr) for attr in required_attrs), \
        f"Training arguments must have attributes {required_attrs}"

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def load_datasets(data_args, model_args) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """
    Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    (the dataset will be downloaded automatically from the datasets Hub)

    For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    behavior (see below)

    In distributed training, the load_dataset function guarantees that only one local process can concurrently
    download the dataset.

    :param data_args: arguments passed to the datasets
    :param model_args: arguments passed to the models
    :return: a tuple of datasets
    """
    if data_args.dataset_name:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
        return raw_datasets

    data_files = {}
    extension = ""
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    assert bool(extension), "No extension provided"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    return raw_datasets


