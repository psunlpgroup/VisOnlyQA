activate_conda_env() {
    local MODEL=$1
    source $CONDA_SH
    
    if [ $MODEL = "xgen-mm-phi3-interleave-r-v1.5" ]
    then
        conda activate xgen-mm
        echo "activated xgen-mm"
    elif [[ $MODEL == *llava_next* ]]
    then
        conda activate llava-next
        echo "activated llava-next"
    elif [[ $MODEL == *Llama-3.2* ]]
    then
        conda activate llama32
        echo "activated llama32"
    elif [[ $MODEL == *molmo* ]]
    then
        conda activate molmo
        echo "activated molmo"
    elif [[ $MODEL == *Phi-3.5-vision-instruct* ]]
    then
        conda activate phi35v
        echo "activated phi35v"
    elif [[ $MODEL == *Qwen2-VL* ]]
    then
        conda activate qwen2vl
        echo "activated qwen2vl"
    elif [[ $MODEL == *InternVL* ]]
    then
        conda activate internvl
        echo "activated internvl"
    else
        conda activate visonlyqa
    fi
}
