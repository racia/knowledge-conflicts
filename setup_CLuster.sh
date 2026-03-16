# conda remove -n kc --all

conda create --name kc1 python=3.11.9

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kc1

conda install nvidia::cuda==12.6.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install accelerate

conda install jupyter notebook -c conda-forge

conda install jsonlines transformers accelerate bitsandbytes>=0.46.1

