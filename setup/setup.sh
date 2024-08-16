source $CONDA_SH

export PYTHONPATH="${PYTHONPATH}:../:../VLMEvalKit:../InternVL/internvl_chat:../alphageometry:../alphageometry/random_generation:./"

conda env create -f setup/environment.yml
conda activate visonlyqa  # main

# for evaluation
# https://github.com/open-compass/VLMEvalKit/commit/9099cdacaa92995fae4ef23c957d1b78d89ae466
git clone https://github.com/open-compass/VLMEvalKit ../VLMEvalKit

pip install torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.37.2 einops==0.8.0 timm==1.0.9 accelerate==0.33.0 sentencepiece==0.2.0

# build another environment for internvl
bash setup/setup_internvl.sh

# build another environment for xgen-mm
bash setup/setup_xgen-mm.sh

# build another environment for phi-3.5-vision
bash setup/setup_phi35v.sh

# build another environment for llama 3.2
bash setup/setup_llama32.sh

# build another environment for llava-next
bash setup/setup_llava-next.sh

# Save model names for VLMEval. This is alreadly included in the repository. You don't need to run it again.
# conda activate llama32
# python src/setup/save_vlmeval_models_list.py
# conda deactivate

#
# for creating train data
# 4b6ebcbc49e3009dad2bff77bcb696243c42ca94
git clone git@github.com:felixludos/alphageometry.git ../alphageometry
conda create -n alphageometry python=3.10.14
conda activate alphageometry
pip install --require-hashes -r ../alphageometry/requirements.txt
pip install typed-argument-parser==1.10.1
