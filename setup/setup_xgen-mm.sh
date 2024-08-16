source $CONDA_SH

conda env create -f setup/environment_xgen-mm.yml
conda activate xgen-mm

pip install -r setup/requirements_vlmevalkit.txt

pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch==2.24.0
pip install einops
pip install einops-exts
pip install transformers==4.41.1

conda deactivate
