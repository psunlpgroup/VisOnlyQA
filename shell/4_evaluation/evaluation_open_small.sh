export MODELS_LIST="Qwen/Qwen2-VL-2B-Instruct Qwen/Qwen2-VL-7B-Instruct microsoft/Phi-3.5-vision-instruct llava_next_llama3 molmo-7B-D-0924 Llama-3.2-11B-Vision-Instruct"
bash shell/4_evaluation/__evaluation.sh real
bash shell/4_evaluation/__evaluation.sh synthetic
unset MODELS_LIST

python src/evaluation/generate_tables.py
