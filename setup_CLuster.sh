# conda remove -n kc --all

conda create --name kc python=3.11.9

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kc

conda install nvidia::cuda==12.8 # not necessary if you have the correct Nvidia drivers installed, but it ensures that the correct CUDA version is available in the environment
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#this version for GTX 1080 Ti Nvidia drivers supports sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 ...
pip3 install accelerate omegaconf

conda install jupyter notebook -c conda-forge
conda install jsonlines transformers

