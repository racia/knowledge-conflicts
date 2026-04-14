# conda remove -n conflicts --all

env_name=conflicts
conda create --name $env_name python=3.11.9 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $env_name

pip3 install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Core libraries
pip3 install "accelerate>=0.20.1,<0.29.0" "numpy<2"
pip3 install jsonlines omegaconf "transformers=4.47.0"

#pip3 install datasets scikit-learn "numpy<2" "huggingf:ace-hub>=0.19.3,<1.0"

# Quantization and memory optimization
pip3 install bitsandbytes==0.42.0
# pip3 install flash-linear-attention==0.3.0

# Optional: Jupyter (via conda)
# conda install jupyter notebook -c conda-forge
