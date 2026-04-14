# conda remove -n torch260 --all

env_name=torch260
conda create --name $env_name python=3.13 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $env_name

pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core libraries
# RWKV doesn't work, and it's an unofficial library
# Kimi gives weird errors out of "transformers>=4.53.0,<5.0.0"
pip3 install "transformers>=4.53.0,<5.0.0" # "transformers=4.48.2"  # "transformers>=4.53.0"
pip3 install "accelerate>=1.0.0" "numpy<2"
pip3 install jsonlines omegaconf scikit-learn
#pip3 install flash-linear-attention==0.3.0
pip3 install -U fla-core

# Quantization and memory optimization
pip3 install bitsandbytes==0.42.0 # 0.49.2

# Optional: Jupyter (via conda)
# conda install jupyter notebook -c conda-forge
