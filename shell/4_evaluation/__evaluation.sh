export REAL_SYNTHETIC=$1

source $CONDA_SH
source shell/4_evaluation/__evaluation_func.sh

for MODEL in $MODELS_LIST
do
    activate_conda_env $MODEL

    # if [ $MODEL = "OpenGVLab/InternVL2-Llama3-76B" ]
    if [ $REAL_SYNTHETIC = "real" ]
    then
        export PROMPTS="no_reasoning reasoning"
    else
        export PROMPTS="no_reasoning"
    fi

    for PROMPT in $PROMPTS
    do
        echo "Evaluating model $MODEL with prompt $PROMPT"
        python src/evaluation/evaluation.py --model $MODEL --prompt $PROMPT --data eval_${REAL_SYNTHETIC}
    done

    conda deactivate
done

conda activate visonlyqa
