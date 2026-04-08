# conda remove -n torch251 --all

env_name=torch251
conda create --name $env_name python=3.11.9 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $env_name

pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#conda install pytorch==2.5.1 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Core libraries
pip3 install "accelerate>=0.20.1,<0.29.0" "numpy<2"
pip3 install jsonlines omegaconf "transformers==4.47.0" scikit-learn

# Quantization and memory optimization
pip3 install bitsandbytes==0.42.0

# Optional: Jupyter (via conda)
# conda install jupyter notebook -c conda-forge
