<h1 align="center"> Planting Knowledge Conflicts into LLM</h1>
<h2 align="center"> Experiments with the MedMCQA Dataset  </h2>

---

This repository shows the work done within a joined project for the following subjects:

- **CrossTemporal NLP**
- **Is Attention All You Need? The Search for a New Architecture**

---

## Table of Contents

- [Authors](#authors)
- [Setup](#setup)
- [Data Cleaning](#data-cleaning)
- [Conflict Planting](#conflict-planting)
- [Paraphrasing Questions](#paraphrasing-questions)
- [Domain Adaptation](#domain-adaptation)

---

## Authors

- Bohdana Ivakhnenko (@bohdana-ivakhnenko)
- Mario Kuzmanov (@MarioKuzmanov-work, @MarioKuzmanov)
- Raziye Sari (@racia)
- Dominik Grosse (@grosse)

---

### Setup

- Create environment, install packages:
  In bash: 
```
cd ~/knowledge-conflicts
source scripts/setup_CLuster.sh
```

--- 

## Data Cleaning

### Running the Script

- Tweak the variables in `start_cleaning.sh`
- Submit a Slurm job:
```
cd ~/knowledge-conflicts
sbatch scripts/start_cleaning.sh
```

---

## Conflict Planting

### Running the Script

- Submit your specific Slurm job: 
```
cd ~/knowledge-conflicts
sbatch scripts/plant_conflicts_<user>.sh
```

---

## Paraphrasing Questions

### Individual Files

The file [`paraphrasing/collector.py`](collector.py) is the main acquisition file.  
In this file there are two main variables, which can be manipulated to achieve desired results.  
`split` is the variable deciding on the dataset split which is to be loaded for paraphrasing and which will be the name
used for the resulting `<split>_questions_and_paraphrases.txt` file.  
The `data` variable is used to load the huggingface dataset one wishes to use.  
The results of the paraphrasing operation will be saved in a text file in the same directory as `collector.py` was
executed in.

The file [`paraphrasing/paraphrase.py`](paraphrase.py) can be used on its own for singular paraphrases using
the [OLlama API](https://ollama.com/library/llama3.2) chat function.  
The variable `SYSTEM_PROMPT` is used to tell the OLlama model what is should do.
By default, the OLlama model used is *llama3.2:3b* since it permits great performance at minimal time investment. This
can be changed by providing a different name when calling the function.
For debugging purposes there is a default question provided in the `question` argument, but it is advised to change
these for actual use.

---

## Domain Adaptation

### Overview

* Model `model_id=google/bigbird-roberta-base`

* Our method uses the transformers library from Hugging Face to fine-tune a MLM head on the target MedMCQA domain, with
  respect to the different context types.

### Usage

1. The dataset is prepared - inspect `data/dataset.ipynb`
    * the final results are in `data/final_data`

2. Run Domain adaptation
    * adjust paths 
    * run locally
    ```
    cd ~/knowledge-conflicts
    bash scripts/run_mlm.sh
    ```
    * run on a Slurm server
    ```
    cd ~/knowledge-conflicts
    sbatch scripts/run_mlm.sh
    ```

3. Run inference
    * `cd domain_adaptation`
    * run the cells in `inference_clean.ipynb` - inference with the model adapted on the baseline contexts
    * run the cells in `inference_conflicts.ipynb` - inference with the model adapted on the mix of baseline and
      conflicting contexts
    * run the cells in `inference_off-the-shelf.ipynb` - inference with the off-the-shelf model without any domain
      adaptation

4. Inspect the results
    * `cd results`
    * ```markdown
      - baseline
      - conflicts
      - off-the-shelf
      ```

* Wherever needed, adjust the paths to input/output files and/or directories
* All outputs are already provided, we however keep the checkpoints of the model locally due to their size

### Domain Adaptation Attention

* model selection available in `models/model_config.yaml` and in `domain_adaptation/model_selection_attn.ipynb`

| Name       | Checkpoint                    | Parameters (M) | Attention Mechanism        | Position Encodings |
|------------|-------------------------------|----------------|----------------------------|--------------------|
| RoBERTa    | FacebookAI/roberta-large      | ~355           | Full                       | Absolute           |
| BigBird    | google/bigbird-roberta-large  | ~360           | Block Sparse               | Absolute           |
| Longformer | allenai/longformer-large-4096 | ~435           | Sliding Window + Global    | Absolute           |
| ModernBERT | answerdotai/ModernBERT-large  | ~396           | Local/Global (Alternating) | RoPE               |

1. The final datasets are available in `data/final_data_attn/`
    - notebook: `data/dataset_attn.ipynb`

2. Domain adaptation related scripts
    - `domain_adaptation/run_mlm_attn.py`
    - `domain_adaptation/run_eval_qa_attn.py`
    - `domain_adaptation/run_mlm_attn.ipynb` # how I ran the script in Colab
    - `domain_adaptation/run_mlm-cleaned.ipynb` # in parallel with same version on the cleaned corpus
    - `scripts/run_mlm_attn.sh`

3. Fine-tuning QA related scripts
   - `case_study/case_study.md` - have a read: [case study](domain_adaptation/case_study/case_study.md)
   - `case_study/fine_tune_qa.ipynb`

4. Results are available in `results/bi_lms`

5. Models are available in `models/bidirectional_attn`

#### Working with large files

* `git lfs install`
* if files are already added to git `git rm --cached` else `git lfs track file/dir name`
    * make sure to include all files when you `git lfs track` directories
* `git add .gitattributes`

**Continue using the files normally**

* `git add`
* `git commit -m`
* `git push`

---