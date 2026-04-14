# conda remove -n rwkv --all

# bash scripts/setup_CLuster_rwkv.sh

env_name=rwkv
conda create --name $env_name python=3.12 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $env_name

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# from the RWKV repository, but it doesn't work, installs transformers 5.5.4 (too new)
# pip3 install -U git+https://github.com/sustcsonglin/flash-linear-attention
pip3 install flash-linear-attention --no-build-isolation --extra-index-url https://pypi.fla-org.com/simple
# pip3 uninstall flash-linear-attention -y

# Core libraries
pip3 install "transformers>=4.35,<4.45" # "transformers>=4.53.0,<5.0.0" was used previously
pip3 install "accelerate>=1.0.0" jsonlines omegaconf scikit-learn datasets

# Quantization and memory optimization
pip3 install bitsandbytes # ==0.42.0 # 0.49.2

# Optional: Jupyter (via conda)
# conda install jupyter notebook -c conda-forge
