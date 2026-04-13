## Case Study

### Domain Adapted models on the QA task

---

* use a domain adapted model to fine-tune on a subset of the MedMCQA dataset and then compare its results versus taking
  off-the-shelf model for fine-tuning.

* **3000** examples of the original **train split** to train, full evaluation on the original **dev set**

* notebook: `fine_tune_qa.ipynb`

* **Limitation** - this is a very small case study, as time is limited, and we would like to see whether domain
  adaptation has or could have impacts on the downstream task already.

---

### Results

| Off-the-shelf    | P-macro % | R-macro % | F1-macro % | Acc % | 
|------------------|-----------|-----------|------------|-------|
| roberta-large    | 23.23     | 23.12     | 22.94      | 23.17 |
| bigbird-large    | 24.80     | 24.79     | 24.58      | 24.79 |
| longformer-large | -         | -         | -          | -     |
| modernBERT-large | 28.32     | 28.44     | 28.01      | 28.11 |

| Original         | P-macro % | R-macro % | F1-macro % | Acc % | 
|------------------|-----------|-----------|------------|-------|
| roberta-large    | 23.49     | 23.56     | 23.49      | 23.88 |
| bigbird-large    | 27.16     | 27.25     | 27.06      | 27.44 |
| longformer-large | -         | -         | -          | -     |
| modernBERT-large | 27.52     | 27.72     | 27.39      | 27.54 |

| Cleaned          | P-macro % | R-macro  % | F1-macro % | Acc % | 
|------------------|-----------|------------|------------|-------|
| roberta-large    | 26.15     | 25.99      | 25.83      | 26.11 |
| bigbird-large    | 28.35     | 28.28      | 28.10      | 28.42 |
| longformer-large | -         | -          | -          | -     |
| modernBERT-large | 27.44     | 27.57      | 27.22      | 27.37 |

* `longformer-large` is excluded from this case study, because it always pads the `input_ids` to a multiple of 512. However, this makes it much more expensive and unnecessary compute as our QA pairs are relatively short.
* generally, it is not designed to be efficient with short sequences.

---


