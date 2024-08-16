export MODELS_LIST="OpenGVLab/InternVL2-4B OpenGVLab/InternVL2-8B OpenGVLab/InternVL2-26B"
bash shell/4_evaluation/__evaluation.sh real
bash shell/4_evaluation/__evaluation.sh synthetic
unset MODELS_LIST

python src/evaluation/generate_tables.py
