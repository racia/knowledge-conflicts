from datasets import load_dataset
from paraphrase import paraphraser
from tqdm import tqdm


split = "validation"
data = load_dataset("openlifescienceai/medmcqa", split=split)


for datapoint in tqdm(data):
    question = datapoint["question"]
    paraphrases = paraphraser(question=question)

    with open(f"{split}_questions_and_paraphrases.txt", "a") as pap:
        pap.write(question)
        pap.write("\n")
        for line in paraphrases:
            pap.write(line)
            pap.write("\n")
