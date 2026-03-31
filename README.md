# Planting Knowledge Conflicts into LLM. Experiments with MedMCQA Dataset.

This repository shows the work done within a joined project for the following subjects:
- CrossTemporal NLP
- Is Attention All You Need? The Search for a New Architecture

---

Contributors of the project:
- Bohdana Ivakhnenko (@bohdana-ivakhnenko)
- Mario Kuzmanov (@MarioKuzmanov-work)
- Raziye Sari (@racia)
- Dominik Grosse (@grosse)

### Set Up the Project
- Create environment, install packages:
In bash: `source setup_CLuster.sh`  


## Data Cleaning

### Running the Script

- Tweak the variables in `start_cleaning.sh`
- Submit a Slurm job: `sbatch start_cleaning.sh`


## Conflict Planting

### Running the Script

- Submit your specific Slurm job: `sbatch plant_conflicts_<user>.sh`


## Paraphrasing Questions

### Individual Files

The file [`paraphrasing/collector.py`](collector.py) is the main acquisition file.  
In this file there are two main variables, which can be manipulated to achieve desired results.  
`split` is the variable deciding on the dataset split which is to be loaded for paraphrasing and which will be the name used for the resulting `<split>_questions_and_paraphrases.txt` file.  
The `data` variable is used to load the huggingface dataset one wishes to use.  
The results of the paraphrasing operation will be saved in a text file in the same directory as `collector.py` was executed in.  

The file [`paraphrasing/paraphrase.py`](paraphrase.py) can be used on its own for singular paraphrases using the [OLlama API](https://ollama.com/library/llama3.2) chat function.  
The variable `SYSTEM_PROMPT` is used to tell the OLlama model what is should do. 
By default, the OLlama model used is *llama3.2:3b* since it permits great performance at minimal time investment. This can be changed by providing a different name when calling the function. 
For debugging purposes there is a default question provided in the `question` argument, but it is advised to change these for actual use.  


## Domain Adaptation

#### `model_id=google/bigbird-roberta-base`

---

### Working with large files

* `git lfs install`
* if files added to git `git rm --cached` else `git lfs track file/dir name`
* `git add .gitattributes`

 Work normally with the files
* `git add`
* `git commit -m`
* `git push`

---