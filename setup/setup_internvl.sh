source $CONDA_SH

git clone git@github.com:OpenGVLab/InternVL.git ../InternVL
cd ../InternVL
git checkout c62fa4f7c850165d7386bdc48ac6bc5a6fab0864
cd internvl_chat

conda create -n internvl python=3.9 -y
conda activate internvl

pip install typed-argument-parser==1.10.1 datasets==2.21.0 Pillow==10.4.0 openai==1.41.1

pip install torch==2.4 torchvision torchaudio
pip install accelerate==0.34.2
pip install peft==0.10.0  # downgrade to avoid transformers issue

pip install -r setup/requirements_internvl.txt

MAX_JOBS=4 pip install flash-attn==2.3.6 --no-build-isolation  # internvl training  # this takes long time

conda deactivate
