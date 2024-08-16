source $CONDA_SH

conda env create -f setup/environment_molmo.yml
conda activate molmo

pip install -r setup/requirements_vlmevalkit.txt

pip install torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install einops==0.8.0

conda deactivate
