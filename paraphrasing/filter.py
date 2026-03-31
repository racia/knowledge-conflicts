from datasets import load_dataset
from evaluate import load
from tqdm import tqdm
import pandas as pd



def evaluate_paraphrases(split:str):
    """
    This function will go through the provided paraphrase file and scann for all original questions.  
    These original questions will then be used as the gold standard for the paraphrases.  
    The results will be saved to a csv.  
    Input files and result files' names will need to be manually adjusted.  

    :param split: str

    :Returns: none
    """
    # Load dataset
    dataset = load_dataset("openlifescienceai/medmcqa", split=split)
    original_questions = set(d["question"] for d in dataset.take(120000))
    # Load paraphrase file
    with open(f"datasets/{split}_questions_and_paraphrases.txt") as f:
        rows = [line.strip() for line in f if line.strip()]
    # Load metrics
    rouge = load("rouge")
    meteor = load("meteor")
    bertscore = load("bertscore")
    # Group paraphrases
    groups = {}
    current_original = None

    for row in rows:
        if row in original_questions:
            current_original = row
            groups[current_original] = []
        else:
            if current_original:
                groups[current_original].append(row)
    results = []

    # Compute metrics
    for qid, (original, paras) in enumerate(tqdm(groups.items())):
        for pid, para in enumerate(paras):

            rouge_result = rouge.compute(
                predictions=[para],
                references=[original]
            )
            meteor_result = meteor.compute(
                predictions=[para],
                references=[original]
            )
            bert_result = bertscore.compute(
                predictions=[para],
                references=[original],
                lang="en"
            )

            results.append({
                "question_id": qid,
                "paraphrase_id": pid,
                "original_question": original,
                "paraphrase": para,
                "rouge1": rouge_result["rouge1"],
                "rouge2": rouge_result["rouge2"],
                "rougeL": rouge_result["rougeL"],
                "meteor": meteor_result["meteor"],
                "bertscore_f1": bert_result["f1"][0]
            })

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(f"{split}_paraphrase_metrics.csv", index=False)

    print("Saved results to paraphrase_metrics.csv")


if __name__ == "__main__":
    #evaluate_paraphrases("validation")
    #evaluate_paraphrases("test")
    evaluate_paraphrases("train")