# Planting Knowledge Conflicts into LLM. Experiments with MedMCQA Dataset.

This repository shows the work done within a joined project for the following subjects:
- CrossTemporal NLP
- Is Attention All You Need? The Search for a New Architecture

Contributors of the project:
- Bohdana Ivakhnenko (@bohdana-ivakhnenko)
- Mario Kuzmanov (@MarioKuzmanov-work)
- Raziye Sari (@racia)
- Dominik Grosse (@grosse)

### Set Up the Project
- Create environment, install packages:
In bash: `source setup_CLuster.sh`  

### Run Data Cleaning Script

- Tweak the variables in `start_cleaning.sh`
- Submit a Slurm job: `sbatch start_cleaning.sh`

### Run Conflict Planting Script

- Submit your specific Slurm job: `sbatch plant_conflicts_<user>.sh`
