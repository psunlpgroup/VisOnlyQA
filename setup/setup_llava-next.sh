source $CONDA_SH

conda env create -f setup/environment_llava-next.yml
conda activate llava-next

pip install -r setup/requirements_vlmevalkit.txt

pip install torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.44.2

conda deactivate
