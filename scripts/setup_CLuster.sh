# conda remove -n kc --all

conda create --name kc python=3.11.9 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kc

pip3 install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Core libraries
pip install transformers==4.38.2
pip install accelerate omegaconf jsonlines safetensors

# Quantization and memory optimization
pip3 install bitsandbytes==0.42.0

# Optional: Jupyter (via conda)
# conda install jupyter notebook -c conda-forge
