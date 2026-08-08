# Script for LLM-as-a-jusge to evaluate pair-wise evidence based on provided metrics, saved in a JSON file.

# Read in data

import csv
import json
import re
from transformers import pipeline

def process_file(file_path):
    with open(file_path, 'r') as csv_f:
        reader = csv.DictReader(csv_f, delimiter="\t")
        header = reader.fieldnames
        data = [row for row in reader]
    return data


def process_evidence(data):
    # Process the evidence data and return a structured format
    processed_data = []
    for row in data:
        processed_row = {
            "Id": row["Id"],
            "Question": row["Question"],
            "Question_upd": row["Question_upd"],
            "Original_exp": row["Original_exp"],
            "Original_ans": row["Original_ans"],
            "Modified_exp": row["Modified_exp"],
            "Modified_ans": row["Modified_ans"],
        }
        processed_data.append(processed_row)
    return processed_data


def build_prompt(evidence_pair):
    base_prompt= """
    You are a judge tasked with comparing two pieces of medical evidence based on the following metrics and decide the superior one:
    1. Relevance: The evidence is relevant and plausible to the answer.
    2. Consistency: The evidence in itself is consistent and does not contradict itself.
    3. Accuracy: The evidence is medically accurate and factually correct.
    Decide for each metric which evidence is better. If there is a tie, output "Equal" respectively. Finally, output the overall better evidence based on the metrics or "Equal" if there is a tie.
    The output format is: 1. Relevance: C{i}, 2. Consistency: C{i}, 3. Accuracy: C{i}, Overall: C{i}.
    Here is the question and evidence pair:
    """
    question = f"Question: {evidence_pair['Question_upd']}"
    base_prompt += f"\n{question}"
    evidence_1 = f"Evidence 1: {evidence_pair['Original_exp']} (Answer: {evidence_pair['Original_ans']})"
    evidence_2 = f"Evidence 2: {evidence_pair['Modified_exp']} (Answer: {evidence_pair['Modified_ans']})"
    built_prompt = f"{base_prompt}\n{evidence_1}\n{evidence_2}"
    
    return built_prompt


def parse_response(response_text):
    # Parse the response text to extract the evaluation results
    lines = re.split(r",\s", response_text)
    results = {}
    for line in lines:
        if line.startswith("1. Relevance:"):
            ev_choice = re.search(r"1\. Relevance:\s*((Evidence|C)\s?\d+|Equal|Both)", line)
            results["Relevance"] = ev_choice.group(1) if ev_choice else line.split(":")[1].strip()
        elif line.startswith("2. Consistency:"):
            ev_choice = re.search(r"2\. Consistency:\s*((Evidence|C)\s?\d+|Equal|Both)", line)
            results["Consistency"] = ev_choice.group(1) if ev_choice else line.split(":")[1].strip()
        elif line.startswith("3. Accuracy:"):
            ev_choice = re.search(r"3\. Accuracy:\s*((Evidence|C)\s?\d+|Equal|Both)", line)
            results["Accuracy"] = ev_choice.group(1) if ev_choice else line.split(":")[1].strip()
        elif line.startswith("Overall:"):
            ev_choice = re.search(r"Overall:\s*((Evidence|C)\s?\d+|Equal|Both)", line) # TODO: Handle invalid index case, i.e. "3" for equal
            results["Overall"] = ev_choice.group(1) if ev_choice else line.split(":")[1].strip()
    
    return results


def save_results_to_json(results, output_file):
    with open(output_file, "w") as json_f:
        json.dump(results, json_f, indent=4)


def evaluate_evidence(evidence_pair):
    built_prompt = build_prompt(evidence_pair)
    pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", trust_remote_code=True)
    messages = [
        {"role": "system", "content": "You are a judge tasked with comparing pair-wise medical evidence."},
        {"role": "user", "content": built_prompt}
    ]
    response = pipe(messages, max_new_tokens=2768, do_sample=False)
    return response[0]['generated_text']


if __name__ == "__main__":
    # Process the CSV file
    file_path = "./data/MedMCQA/evidence_data.csv"
    data = process_file(file_path)
    processed_data = process_evidence(data)

    # Evaluate each evidence pair and save results
    results = []
    for evidence_pair in processed_data[:10]:  # Limit to first 10 pairs for testing
        response_text = evaluate_evidence(evidence_pair)
        response_block = re.sub(r'<think>.*?</think>', '', response_text[-1]["content"], flags=re.DOTALL)
        print(f"Response for evidence pair ID {evidence_pair['Id']}:\n{response_text}\n")
        evaluation_results = parse_response(response_block)
        evaluation_results["Id"] = evidence_pair["Id"]
        results.append(evaluation_results)

    # Save the results to a JSON file
    output_file = "./data/MedMCQA/evidence_evaluation_results.json"
    save_results_to_json(results, output_file)