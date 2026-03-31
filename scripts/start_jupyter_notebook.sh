#!/usr/bin/env bash
#
#SBATCH --job-name=jupyter_test
#SBATCH --output=jupyter_test_out
#SBATCH --error=jupyter_test_err
#SBATCH --partition=students
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu09
#SBATCH --mail-user=ivakhnenko@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

# ALTERNATIVE COMMAND TO START INTERACTIVE JOB
# salloc -p students -n 1 --mem=128000 --gres=gpu:1

# JOB STEPS ON CLUSTER
srun hostname
# to get the same from python:
# import socket; print(socket.gethostname())

conda activate kc
srun jupyter notebook --no-browser --ip 0.0.0.0 --port 8888

# LOCALLY
# Explanation:
# ssh -L (local jupyter port):(srun hostname):(check in the output, like http://gpu09:8889/tree) USERNAME@cluster.cl.uni-heidelberg.de
# Example:
# ssh -L 8888:gpu08:8888 ivakhnenko@cluster.cl.uni-heidelberg.de

# Then open in local browser:
# http://localhost:8888
# 8888 stays local (the first one)
