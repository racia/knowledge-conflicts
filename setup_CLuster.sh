# conda remove -n kc --all

conda create --name kc1 python=3.11.9

conda activate kc

conda install nvidia::cuda==12.6.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install accelerate

conda install jupyter notebook -c conda-forge

conda install jsonlines transformers accelerate

