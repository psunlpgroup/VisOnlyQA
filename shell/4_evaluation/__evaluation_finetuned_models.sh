source $CONDA_SH
source shell/4_evaluation/__evaluation_func.sh

activate_conda_env $MODEL

for PROMPT in no_reasoning
do
    echo "Evaluating model $MODEL with prompt $PROMPT on Real Eval Data"
    python src/evaluation/evaluation.py --model $MODEL --prompt $PROMPT --data eval_real
    python src/evaluation/bootstrap.py --model $MODEL --prompt $PROMPT --data eval_real

    echo "Evaluating model $MODEL with prompt $PROMPT on Synthetic Eval Data"
    python src/evaluation/evaluation.py --model $MODEL --prompt $PROMPT --data eval_synthetic
    python src/evaluation/bootstrap.py --model $MODEL --prompt $PROMPT --data eval_synthetic
done

conda deactivate

python src/evaluation/generate_tables_finetuned.py
