## Case Study

### Domain Adapted models on the QA task

---

* use a domain adapted model to fine-tune on a subset of the MedMCQA dataset and then compare its results versus taking
  off-the-shelf model for fine-tuning.

* _ examples of the original train split to train, full evaluation on the original dev set

* notebook: `fine_tune_qa.ipynb`

* **Limitation** - this is a very small case study, as time is limited and we would like to see whether domain adaptation has or could have impacts on the downstream task already

---

### Results

| Off-the-shelf    | P-macro | R-macro | F1-macro | Acc | 
|------------------|---------|---------|----------|-----|
| roberta-large    |         |         |          |     |
| bigbird-large    |         |         |          |     |
| longformer-large |         |         |          |     |
| modernBERT-large |         |         |          |     |

| Original         | P-macro | R-macro | F1-macro | Acc | 
|------------------|---------|---------|----------|-----|
| roberta-large    |         |         |          |     |
| bigbird-large    |         |         |          |     |
| longformer-large |         |         |          |     |
| modernBERT-large |         |         |          |     |


| Cleaned          | P-macro | R-macro | F1-macro | Acc | 
|------------------|---------|---------|----------|-----|
| roberta-large    |         |         |          |     |
| bigbird-large    |         |         |          |     |
| longformer-large |         |         |          |     |
| modernBERT-large |         |         |          |     |


---


