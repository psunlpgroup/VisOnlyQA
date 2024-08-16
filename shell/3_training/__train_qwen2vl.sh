source $CONDA_SH
conda activate qwen2vl
cd ../LLaMA-Factory

# Our parameters assume to use four GPUs. If you train on a different number of GPUs, please update the batch number configurations.
CUDA_VISIBLE_DEVICES=${GPUS} llamafactory-cli train ../VisOnlyQA/config/qwen2vl_train_yaml/Qwen2-VL-${MODEL_SIZE}B-Instruct/${SPLIT}.yaml

cd ../VisOnlyQA
conda deactivate
