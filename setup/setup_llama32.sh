source $CONDA_SH

conda env create -f setup/environment_llama32.yml
conda activate llama32

pip install -r setup/requirements_vlmevalkit.txt

pip install torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.45.0

conda deactivate
