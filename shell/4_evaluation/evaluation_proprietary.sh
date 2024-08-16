for MODEL in claude-3-5-sonnet-20240620 gpt-4o-mini-2024-07-18 gpt-4o-2024-08-06 gemini-1.5-flash-002 gemini-1.5-pro-002
do
    for DATA in real synthetic
    do
        for PROMPT in no_reasoning reasoning
        do
            # skip synthetic and reasoning
            if [ $DATA = "synthetic" ] && [ $PROMPT = "reasoning" ]
            then
                continue
            fi

            echo "Evaluating model $MODEL on $DATA data with prompt $PROMPT"
            python src/evaluation/evaluation.py --model $MODEL --prompt $PROMPT --data eval_${DATA}
        done
    done
done

python src/evaluation/generate_tables.py
