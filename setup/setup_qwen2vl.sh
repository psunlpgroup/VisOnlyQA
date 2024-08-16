source $CONDA_SH

conda env create -f setup/environment_qwen2vl.yml
conda activate qwen2vl

pip install torch==2.5 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.46.1
pip install qwen-vl-utils==0.0.8

# https://github.com/hiyouga/LLaMA-Factory
pip install accelerate==0.32.0 peft==0.12.0 trl==0.9.6 bitsandbytes==0.43.1 vllm==0.5.0
pip install deepspeed==0.15.4
MAX_JOBS=4 pip install flash-attn==2.6.3 --no-build-isolation

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git ../LLaMA-Factory
cd ../LLaMA-Factory
pip install -e ".[torch,metrics]"

conda deactivate
