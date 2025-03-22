export MODELS_LIST="llava_next_yi_34b Qwen/Qwen2-VL-72B-Instruct molmo-72B-0924 Llama-3.2-90B-Vision-Instruct"
bash shell/4_evaluation/__evaluation.sh real
bash shell/4_evaluation/__evaluation.sh synthetic
unset MODELS_LIST

python src/evaluation/generate_tables.py
