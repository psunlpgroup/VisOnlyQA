source $CONDA_SH

conda env create -f setup/environment_phi35v.yml
conda activate phi35v

# pip install -r setup/requirements_vlmevalkit.txt

pip install numpy==1.24.4 Pillow==10.3.0 Requests==2.31.0 torch==2.3.0 torchvision==0.18.0 transformers==4.43.0 accelerate==0.30.0 # flash_attn==2.5.8
pip install accelerate==0.30.1 peft deepspeed==0.14.4  # phi-3.5 training

conda deactivate
