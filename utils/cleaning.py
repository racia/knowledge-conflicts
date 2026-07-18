import json
import re

from jsonlines import jsonlines
from pathlib import Path

from utils.model import ModelLoader


class DataCleaner:
    model_loader = None

    def __init__(self):
        self.pruned_exp = 0
        self.cleaned_exp = 0

    def set_model_loader(self, model_loader: ModelLoader):
        self.model_loader = model_loader

    def reset_pruned_exp(self):
        self.pruned_exp = 0

    def clean_exp_with_cop(self, exp: str = None, cop: str = None):
        """
        Prunes the provided str explanation of correct answer exposures from the provided cop answer.
        :param exp: The explanation string to be pruned
        :param cop: The corresponding correct answer string
        :return: If matched, an answer-pruned explanation, otherwise original is returned
        """
        print("Original explanation:", exp, "with cop:", cop)
        assert cop not in ("", None), f"Detected empty-string cop, consider adjusting regex pattern."
        ans_exp_pat = r"^(Ans[.:]?|Answer-?)\s*(?:\s*is\.?\s*)?\s*\(?'?[A-Za-z]'?\)?"
        ans_exp_match = re.match(fr"{ans_exp_pat}(.*?){cop}", exp, re.DOTALL)
        if ans_exp_match:
            # print("Matched string: ", ans_exp_match.string)
            exp_new = exp.replace(ans_exp_match.group(0), "").lstrip()
            # print(f"Successfully cleaned exp: {exp_new}")
            self.pruned_exp += 1
        return exp_new if ans_exp_match else exp

    @staticmethod
    def clean_data(self, base_dir: str, source: str, split: str, prompt: dict, task: str, model, tokenizer) -> None:
        cleaned_path = Path("utils/cleaned_data.jsonl")
        if cleaned_path.exists() and cleaned_path.stat().st_size > 0:
            with jsonlines.open(cleaned_path, "r") as existing_f:
                existing_entries = [entry for entry in existing_f]
                print(f"Found existing cleaned data with {len(existing_entries)} entries...")
                for j, entry in enumerate(existing_entries):
                    if "exp_upd" not in entry:
                        print(f"Target field 'exp_up' missing in entry {j} of existing cleaned data. Starting from this entry index: {j}")
                        # print(json.dumps(entry, indent=4))
                        self.cleaned_exp = j
                        break
                        
            

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
        if task == "question":
            print(entry["question"])
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


    def process_split(self, base_dir: str, source: str, split: str, prompt: dict, task, model, tokenizer) -> None:
        """
        Update the entries of a data split with fixed questions generated by the model.

        :param path: The base path to the dataset (from the point where the main script is being run).
        :param source: The subdirectory under the base path where the original data is located.
        :param split: The data split to process (e.g., "train", "dev, "test").
        :param prompt: The system prompt to use for generation, already formatted with examples.
        :param task: The specific cleaning task to perform ("question" or "explanation").
        :param model: The language model to use for generation.
        :param tokenizer: The tokenizer corresponding to the model.
        """
        if "cleaned" in source:
            source_path = Path(f"{base_dir}/{source}/{split}_classification.jsonl")
        else:
            source_path = Path(f"{base_dir}/{source}/{split}.jsonl")
        with jsonlines.open(source_path, "r") as f:
            data_split = [entry for entry in f]

        target = {
            "question": "question_upd", # 115697
            "explanation": "exp_upd", # TODO: change when classification task is done
            "classification": "exp_to_edit", # 146494
        }[task]
        cleaned_path = Path(f"{base_dir}/cleaned/{split}_{task}.jsonl")
        print("Saving cleaned data to:", cleaned_path)
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)

        start = 0
        existing_entries = []
        if cleaned_path.exists() and cleaned_path.stat().st_size > 0:
            with jsonlines.open(cleaned_path, "r") as existing_f:
                existing_entries = [entry for entry in existing_f]
                # start = len(existing_entries)
                print(f"Found existing cleaned data with {len(existing_entries)} entries...")
                for j, entry in enumerate(existing_entries):
                    if target in entry:
                        start += 1
                    else:
                        print(f"Target field '{target}' missing in entry {j} of existing cleaned data. Starting from this entry index: {j}")
                        print(json.dumps(entry, indent=4))
                        break
            upd_data_json = jsonlines.open(cleaned_path, "a", flush=True)
        else:
            upd_data_json = jsonlines.open(cleaned_path, "w", flush=True)

        answer_in_original_q = 0
        answer_in_edited_q = 0
        comments = 0

        print(f"Starting processing '{split}' from entry index:", start)

        if existing_entries and start:
            for i in range(start):
                question_same = data_split[i]["question"] == existing_entries[i]["question"]
                i_same = data_split[i]["i"] == existing_entries[i]["i"]
                assert question_same and i_same, \
                    (f"Data mismatch at index {i} between original and existing cleaned data.\n"
                    f"Original question {data_split[i]['i']}: {data_split[i]['question']}\n"
                    f"Existing cleaned question {existing_entries[i]['i']}: {existing_entries[i]['question']}")

        try:
            for i, entry in enumerate(data_split[start:], start=start):
                print(f"[{split}] Entry {i}:")
                existing_entry = existing_entries[i] if len(existing_entries) > i else None
                if existing_entry and target in existing_entry:
                    print(f"Entry {i} already processed, just writing to file...")
                    upd_data_json.write(data_split[i])
                    continue

                if "exp" in target and entry["exp"] and len(entry["exp"]) > 8000:
                    print("Original explanation length:", len(entry["exp"]))
                    print(
                        "Original explanation is very long, skipping length check for the updated explanation.")
                    data_split[i]["exp_to_edit"] = None
                    data_split[i]["exp_upd"] = None
                    upd_data_json.write(data_split[i])
                    continue  # train 6882 is a problematic example

                inputs = self.prepare_entry(entry, task)
                if task == "question":
                    answer_in_orig_q = any(a.lower() in inputs["question"].lower() for a in inputs["answer_options"])
                    data_split[i]["op_in_question"] = answer_in_orig_q
                    answer_in_original_q += int(answer_in_orig_q)
                elif task in ["explanation", "classification"] and not inputs["explanation"]:
                    print("No explanation provided, skipping...", end="\n\n")
                    data_split[i]["exp_upd"] = None
                    data_split[i]["exp_to_edit"] = None
                    upd_data_json.write(data_split[i])
                    continue

                chat = self.model_loader.format_chat(prompt, tokenizer, task=task, **inputs)

                output = ""
                if task == "question":
                    answer_in_upd_q, comment_present = True, True
                    iteration = 0
                    len_diff = True
                    while (answer_in_upd_q or comment_present or len_diff) and iteration < 10:
                        iteration += 1
                        output = self.model_loader.generate_text(model, tokenizer, chat, temperature=0.3)
                        len_diff = (self.check_drastic_length_diff(inputs["question"], output)
                                    and len(output) > 100)

                        answer_in_upd_q = any(a in output for a in inputs["answer_options"])
                        answer_in_edited_q += answer_in_upd_q
                        if answer_in_upd_q:
                            print(f"Prompt-leaking is detected, iteration {iteration}: {output}")

                        comment_present = "\n" in output
                        comments += comment_present
                        if comment_present:
                            print(f"Comment is detected, iteration {iteration}: {output}")

                    data_split[i]["question_upd"] = output
                    data_split[i]["op_in_question_upd"] = answer_in_upd_q
                    data_split[i]["comment_in_question_upd"] = comment_present

                elif task == "explanation":
                    if not data_split[i]["exp_to_edit"] or len(data_split[i]["exp"]) < 100:
                        print("Explanation doesn't need to be edited, skipping...", end="\n\n")
                        data_split[i]["exp_upd"] = None
                        data_split[i]["comment_in_exp_upd"] = None
                        upd_data_json.write(data_split[i])
                        continue
                    comment_present, len_diff = True, True
                    iteration = 0
                    while (comment_present or len_diff) and iteration < 5:
                        iteration += 1
                        output = self.model_loader.generate_text(model, tokenizer, chat, temperature=0.3)
                        len_diff = (self.process_splitcheck_drastic_length_diff(inputs["explanation"], output)
                                    and len(output) > 100)
                        comment_present = "\n" in output and "formatted" in output
                        comments += comment_present
                        if comment_present:
                            print(f"Comment is detected, iteration {iteration}: {output}")

                    data_split[i]["exp_upd"] = output
                    data_split[i]["comment_in_exp_upd"] = comment_present

                elif task == "classification":
                    iteration = 0
                    init_output = self.model_loader.generate_text(model, tokenizer, chat)
                    output = self.validate_classification_output(init_output)
                    while output is None and i < 10:
                        iteration += 1
                        print(f"Invalid classification output, regenerating: {output}")
                        init_output = self.model_loader.generate_text(model, tokenizer, chat)
                        print("Raw output:", init_output)
                        output = self.validate_classification_output(init_output)

                    data_split[i]["exp_to_edit"] = output

                print(output, end="\n\n")

                upd_data_json.write(data_split[i])

            print("Answer options in the original question:", answer_in_original_q)
            print("Answer options in the updated question:", answer_in_edited_q)
            print("Cases when the model possibly added a comment:", comments)

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

