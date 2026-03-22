import re

from jsonlines import jsonlines
from pathlib import Path

from utils.model import format_chat, generate_text


def check_drastic_length_diff(
        original: str | int, edited: str | int, threshold: float = 2.0
        ) -> bool:
    """
    Checks if one string is at least `threshold` times longer than the other.

    Args:
        original: Original question string.
        edited: Rewritten question string.
        threshold: Minimum ratio for "drastic" (default 2.0).

    Returns:
        True if drastic difference detected.

    Example:
        >>> check_drastic_length_diff("a"*465, "b"*68)
        True  # 465/68 ≈ 6.8 > 2.0
        >>> check_drastic_length_diff("abc", "abcd")
        False  # 4/3 ≈ 1.33 < 2.0
    """
    len_orig = original if type(original) is int else len(original)
    len_edit = edited if type(edited) is int else len(edited)
    if len_orig == 0 or len_edit == 0:
        return True  # Treat empty as drastic
    ratio = max(len_orig / len_edit, len_edit / len_orig)
    return ratio >= threshold


def prepare_entry(entry: dict, task: str) -> dict:
    """
    Prepare a single data entry for the specified cleaning task.

    :param entry: The original data entry, containing the question and answer options.
    :param task: The cleaning task to perform ("question" or "explanation").
    :return: A dictionary with the formatted input for the model.
    """
    if task in ["question", "answer"]:
        return {
            "question": entry["question"],
            "answer_options": [entry["opa"], entry["opb"], entry["opc"], entry["opd"]],
        }
    elif task in ["explanation", "classification"]:
        print(entry["exp"])
        return {
            "explanation": entry.get("exp", ""),
        }
    else:
        raise ValueError(f"Unknown task: {task}")


def one_letter_options(answer_options: list[str]) -> bool:
    """
    Check if all answer options are single letters (e.g., "a", "b", "c", "d", "e").
    :param answer_options: A list of answer option strings to check.
    :return: True if all answer options are single letters, False otherwise.
    """
    return all(re.fullmatch(r"[abcdef]", op) for op in answer_options)


def only_option_letters(answer_options: list[str]) -> bool:
    """
    Check if all answer options consist solely of single letters (e.g., "a", "b", "c", "d", "e") without any additional text.
    :param answer_options: A list of answer option strings to check.
    :return: True if all answer options are exactly single letters, False otherwise.
    """
    return all(re.fullmatch(r"(?i)[abcdef]+", op.strip()) for op in answer_options)


def ans_op_present(question: str, answer_options: list[str]) -> list[str]:
    """
    Check if any of the answer options are present in the updated question.
    :param question: The updated question string to check.
    :param answer_options: A list of answer option strings to look for in the question.
    :return: True if any of the answer options are present in the updated question, False otherwise
    """
    if one_letter_options(answer_options):
        answers = [f"{re.escape(option)}\b" for option in answer_options]
    else:
        answers = [f"\b{re.escape(option)}\b" for option in answer_options]
    joined_answers = "|".join(answers)
    found_answers = re.findall(rf'{joined_answers}', question)
    return found_answers


def op_list_present(question: str) -> bool:
    """
    Check if any of the answer options are present in the updated question.
    :param question: The updated question string to check.
    :return: True if any of the answer options are present in the updated question, False otherwise
    """
    if re.search(rf'a\)[\s\S]+b\)[\s\S]+c\)[\s\S]+d\)', question):
        return True
    return False


def validate_classification_output(output: str) -> bool | None:
    """
    Validate the model's output for a classification task, ensuring it is either "true" or "false".
    :param output: The raw output from the model that needs to be validated.
    :return: True if the output indicates "true", False if it indicates "false", or None if it's invalid.
    """
    if "true" in output.lower():
        return True
    if "false" in output.lower():
        return False
    else:
        return None


def comment_present(output: str) -> bool:
    """
    Check if the model's output contains a comment or explanation, which may indicate that the model is not following instructions properly.
    :param output: The raw output from the model that needs to be checked for comments.
    :return: True if a comment is detected, False otherwise.
    """
    if not output:
        return False
    suspicious_phrases = [
        "rewritten text", "original text", "Note: I", "answer choices",
        "reformatted the text", "Take a deep breath", "formatted", "I cannot",
        "anything else I can help you with?", "explore the concept",
    ]
    output = output.split("\n\n")
    filtered_output = []
    for part in output:
        if not any(phrase in part for phrase in suspicious_phrases):
            filtered_output.append(part)
    return len(filtered_output) != len(output)


def remove_comment(exp: str) -> str:
    exp = exp.split("\n\n")
    suspicious_phrases = [
        "Here is", "rewritten text", "original text", "Note: I", "answer choices",
        "reformatted the text", "Take a deep breath", "anything else I can help you with?",
        "explore the concept", "I cannot",
    ]
    filtered_exp = []
    for part in exp:
        if not any(phrase in part for phrase in suspicious_phrases):
            filtered_exp.append(part)
    if not filtered_exp:
        print(f"Explanation contains only suspicious phrases: {exp}")
    return "\n\n".join(filtered_exp)


STATISTICS = {
    "question": {
        "op_list_in_question": 0,
        "op_list_in_question_upd": 0,
        "ans_op_in_question_upd": 0,
        "comment_in_question_upd": 0,
    },
    "answer": {
        "comment_in_answer_upd": 0
    },
    "explanation": {
        "comment_in_exp_upd": 0,
    },
    "classification": {}
}


def check_output(task: str, output: str, inputs: dict) -> dict:
    """
    Determine if the model's output indicates that regeneration is needed based
    on the specified task and the presence of answer options or comments.
    :param task: The cleaning task being performed ("question" or "explanation").
    :param output: The raw output from the model that needs to be evaluated for potential issues.
    :param inputs: A dictionary with the formatted input for the model.
    :return:
    """
    if task != "classification":
        if comment_present(output):
            output = remove_comment(output)
            if output:
                if task == "question":
                    STATISTICS[task]["comment_in_question_upd"] += 1
                elif task == "explanation":
                    STATISTICS[task]["comment_in_exp_upd"] += 1
                elif task == "answer":
                    STATISTICS[task]["comment_in_answer_upd"] += 1
                comment = False # no need to fix anymore
            else:
                comment = True
                if task == "question":
                    STATISTICS[task]["comment_in_question_upd"] += comment
                elif task == "explanation":
                    STATISTICS[task]["comment_in_exp_upd"] += comment
                elif task == "answer":
                    STATISTICS[task]["comment_in_answer_upd"] += comment
        else:
            comment = False

        if comment:
            print(f"Comment is detected: {output}")
    else:
        comment = False

    if task == "question":
        if not inputs or any(i not in inputs for i in ["question", "answer_options"]):
            raise ValueError("For 'question' task, 'inputs' must contain 'question' and 'answer_options'.")

        op_list_in_q_orig = op_list_present(inputs["question"])
        op_letters_in_upd_q = op_list_present(output)
        STATISTICS[task]["op_list_in_question"] += int(op_list_in_q_orig)
        STATISTICS[task]["op_list_in_question_upd"] += int(op_letters_in_upd_q)

        op_in_q_disappeared = False
        if op_list_in_q_orig and not op_letters_in_upd_q:
            print("Answer options list disappeared from the question.")
            op_in_q_disappeared = True

        ans_op_in_q = ans_op_present(inputs["question"], inputs["answer_options"])
        ans_op_in_q_upd = ans_op_present(output, inputs["answer_options"])
        ans_op_diff = ans_op_in_q != ans_op_in_q_upd
        ans_op_len_diff = len(ans_op_in_q) != len(ans_op_in_q_upd)
        more_ans_op_in_q_upd = False
        if ans_op_diff and ans_op_len_diff:
            all_op_in_q_upd = len(ans_op_in_q_upd) == len(inputs['answer_options'])
            if len(ans_op_in_q_upd) > len(ans_op_in_q) and not all_op_in_q_upd:
                STATISTICS[task]["ans_op_in_question_upd"] += 1
                print(f"Prompt-leaking is detected for answer options: {inputs['answer_options']}")
                more_ans_op_in_q_upd = True
            elif len(ans_op_in_q) > len(ans_op_in_q_upd):
                print(f"Answer options disappeared from the question: {inputs['answer_options']}")
                op_in_q_disappeared = True

        length_diff = len(output) > 100 and check_drastic_length_diff(inputs["question"], output)

        result = {
            "question_upd": output,
            "op_list_in_question": op_list_in_q_orig,
            "op_list_in_question_upd": op_letters_in_upd_q,
            "op_in_q_disappeared": op_in_q_disappeared,
            "ans_op_in_q_upd": more_ans_op_in_q_upd,
            "length_diff": length_diff,
            "comment_in_question_upd": comment,
        }

    elif task == "answer":
        answer_options = re.split(r'\n', output)
        options = ["A", "B", "C", "D"]
        answer_option_map: dict[str, str | None] = {
            "opa_upd": None,
            "opb_upd": None,
            "opc_upd": None,
            "opd_upd": None,
        }
        if len(answer_options) == 4:
            for op, ans in zip(options, answer_options):
                pattern = re.compile(rf"{op}.")
                if pattern.match(ans.strip()):
                    answer_option_map[f"op{op.lower()}_upd"] = pattern.sub("", ans, count=1).strip()
                else:
                    print(f"Answer option '{op}' is not properly formatted in the output: '{ans.strip()}'")
                    answer_option_map[f"op{op.lower()}_upd"] = None

        else:
            print(f"Expected 4 answer options in the output, but got {len(answer_options)}. Output:\n{output}")

        result = {
            **answer_option_map,
            "comment_in_answer_upd": comment,
        }

    elif task == "explanation":
        length_diff = len(output) > 100 and check_drastic_length_diff(inputs["explanation"], output)
        result = {
            "exp_upd": output,
            "length_diff": length_diff,
            "comment_in_exp_upd": comment,
        }

    elif task == "classification":
        result = {
            "exp_to_edit": validate_classification_output(output),
        }

    else:
        raise ValueError(f"Unknown task: {task}")

    return result


def regeneration_needed(task: str, check_result: dict) -> bool:
    """
    Determine if regeneration is needed based on the check results for the specified task.
    :param task: The cleaning task being performed ("question" or "explanation").
    :param check_result: A dictionary containing the results of checks performed on the model's output.
    :return: True if regeneration is needed, False otherwise.
    """
    if task == "question":
        return (
            check_result["op_in_q_disappeared"] or
            check_result["ans_op_in_q_upd"] or
            check_result["length_diff"] or
            check_result["comment_in_question_upd"]
        )
    elif task == "answer":
        all_answers = all([v for k, v in check_result.items() if k.startswith("op")])
        return not all_answers or check_result["comment_in_answer_upd"]
    elif task == "explanation":
        return check_result["comment_in_exp_upd"]
    elif task == "classification":
        return check_result["exp_to_edit"] is None
    else:
        raise ValueError(f"Unknown task: {task}")


def process_split(
        base_dir: str,
        source: str,
        split: str,
        prompt: dict,
        task,
        model,
        tokenizer,
        reverse: bool = False,
        filtering_ids: str = None
) -> None:
    """
    Update the entries of a data split with fixed questions generated by the model.

    :param base_dir: The base path to the dataset (from the point where the main script is being run).
    :param source: The subdirectory under the base path where the original data is located.
    :param split: The data split to process (e.g., "train", "dev, "test").
    :param prompt: The system prompt to use for generation, already formatted with examples.
    :param task: The specific cleaning task to perform ("question" or "explanation").
    :param model: The language model to use for generation.
    :param tokenizer: The tokenizer corresponding to the model.
    :param reverse: Whether to process the data in reverse order (default: False).
    :param filtering_ids: Optional name of the file containing entry IDs to filter and process (e.g., "empty_ids").
    :return: None
    """
    if "cleaned" in source:
        source_path = Path(f"{base_dir}/{source}/{split}_classification.jsonl")
    else:
        source_path = Path(f"{base_dir}/{source}/{split}.jsonl")
    print("Loading data from:", source_path)
    with jsonlines.open(source_path, "r") as f:
        data_split = [entry for entry in f]

    target = {
        "question": ["question_upd"],
        "classification": ["exp_to_edit"],
        "explanation": ["exp_upd"],
        "answer": ["opa_upd", "opb_upd", "opc_upd", "opd_upd"],
    }[task]
    cleaned_path_straight = Path(f"{base_dir}/cleaned/{split}_{task}.jsonl")
    cleaned_path_reverse = Path(f"{base_dir}/cleaned/{split}_{task}_reverse.jsonl")
    cleaned_path_missing = Path(f"{base_dir}/cleaned/{split}_{task}_{filtering_ids}.jsonl")
    cleaned_path_missing_reverse = Path(f"{base_dir}/cleaned/{split}_{task}_{filtering_ids}_reverse.jsonl")

    existing_entries = {}
    existing_entries_filtered = {}
    all_cleaned_paths = [
        cleaned_path_straight, cleaned_path_reverse,
        cleaned_path_missing, cleaned_path_missing_reverse
    ]
    for path in all_cleaned_paths:
        if path.exists() and path.stat().st_size > 0:
            with jsonlines.open(path, "r") as existing_f:
                entries = {e["i"]: e for e in [e for e in existing_f] if any([t in e for t in target])}
                print(f"Found existing cleaned data in {path} with {len(entries)} entries...")
                existing_entries.update(entries)
                if "missing" in path.name:
                    existing_entries_filtered.update(entries)

    if filtering_ids and reverse:
        output_path = cleaned_path_missing_reverse
    elif filtering_ids:
        output_path = cleaned_path_missing
    elif reverse:
        output_path = cleaned_path_reverse
    else:
        output_path = cleaned_path_straight

    print("Saving cleaned data to:", output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        upd_data_json = jsonlines.open(output_path, "a", flush=True)
    else:
        upd_data_json = jsonlines.open(output_path, "w", flush=True)

    if filtering_ids:
        path = Path(f"data/MedMCQA/{filtering_ids}.txt")
        if not path.exists():
            raise FileNotFoundError(f"Filtering IDs file not found: {path}")
        with open(path, "r") as f:
            filtering_ids = list(map(int, [line for line in f.read().splitlines() if not line.startswith("#") and line.strip()]))
            print(f"Loaded {len(filtering_ids)} IDs for filtering from {path}")
            print(f"Sample empty IDs of length {len(filtering_ids)}:", filtering_ids[:10])
    filtering_ids = filtering_ids or []

    data = data_split[::-1] if reverse else data_split
    data = {e["i"]: e for e in data}
    print("Total entries in the original data split:", len(data))

    try:
        for i, entry in data.items():
            if filtering_ids and i not in filtering_ids:
                continue

            if i == 79613:
                print("CUDA out of memory issue with entry 79613, skipping:", entry)
                continue

            if i in existing_entries and all([t in existing_entries[i] for t in target]):
                already_generated = i in filtering_ids and i in existing_entries_filtered
                if already_generated or not filtering_ids:
                    print(f"Entry {i} already processed, skipping...")
                    continue

            if task in ["classification", "explanation"] and entry["exp"] and not 100 < len(entry["exp"]) < 8000:
                    print(f"Entry {i} contains an explanation that is too short or long, adding None's and skipping...")
                    data[i]["exp_to_edit"] = None
                    data[i]["exp_upd"] = None
                    data[i]["comment_in_exp_upd"] = None
                    upd_data_json.write(data[i])
                    continue

            if task == "explanation" and not entry.get("exp_to_edit", None):
                print(f"Entry {i} doesn't require an explanation update, skipping...")
                data[i]["exp_upd"] = None
                data[i]["comment_in_exp_upd"] = None
                upd_data_json.write(data[i])
                continue

            print(f"[{split}] Entry {i}:")

            inputs = prepare_entry(entry, task)
            if task in ["classification", "explanation"] and entry["exp"] and not inputs["explanation"]:
                raise ValueError("Explanation field is present but empty after preparation:", inputs)

            if task == "answer" and only_option_letters(inputs["answer_options"]):
                print(f"Entry {i} has answer options that are only letters, which may lead to prompt-leaking. Skipping...")
                data[i]["opa_upd"] = None
                data[i]["opb_upd"] = None
                data[i]["opc_upd"] = None
                data[i]["opd_upd"] = None
                data[i]["comment_in_answer_upd"] = False
                upd_data_json.write(data[i])
                continue

            chat = format_chat(prompt, tokenizer, task=task, **inputs, no_ans_op=False)

            output = None
            iteration = 0
            result_dict = {}
            to_generate = True
            while (output is None or to_generate) and iteration < 10:
                iteration += 1
                print(f"Iteration {iteration}:")
                output = generate_text(model, tokenizer, chat, temperature=0.2)
                result_dict = check_output(task, output, inputs)
                print("Check result:", result_dict)

                output = [result_dict[t] for t in target if t in result_dict]
                print(output, end="\n\n")

                to_generate = regeneration_needed(task, result_dict)

            data[i].update(result_dict)
            upd_data_json.write(data[i])

        print(f"Finished processing split {split} for task {task}. Final statistics:")
        for key, stat in STATISTICS[task].items():
            print(f"{key}: {stat}")

        upd_data_json.close()

    except Exception as e:
        upd_data_json.close()
        print(f"An error occurred: {e}")
        raise e

    #     Entry 20:
    # 2, 3-BPG binds to sites of haemoglobin and the affinity for oxygen
    # What binds to sites of haemoglobin and affects the affinity for oxygen?

    # (Note: I assumed the correct question word is "What" since it's a common and logical choice.)

    # Entry 40:
    # Which of the following is not. true regarding myelopathy?
    # What is true regarding myelopathy?

    # Entry 46:
    # Which pa of brachial plexus do not give branches
    # What part of the brachial plexus does not give rise to branches?

    # (Note: I removed the answer options as per the guidelines)


if __name__ == "__main__":
    s = ("Here is the rewritten text:\n\n"
         "Ranitidine is a drug that can cause gynecomastia. It's essential to be aware of the various "
         "medications that can lead to this condition, as this topic has been discussed extensively.")
    print("Comment detected:", comment_present(s))
    q = "Characteristics of Remifentanyl – a) Metabolised by plasma esteraseb) Short half lifec) More potent than Alfentanyld) Dose reduced in hepatic and renal diseasee) Duration of action more than Alfentanyl"
    q_upd = "What are the characteristics of remifentanil, including its metabolism, potency, and duration of action, compared to alfentanil?"
    print("Answer options present in original question:", op_list_present(q))
    print("Answer options present in updated question:", op_list_present(q_upd))

