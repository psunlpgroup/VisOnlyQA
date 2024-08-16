source $CONDA_SH
conda activate phi35v

first_device=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
export MASTER_PORT=$((34229 + $first_device))
echo "MASTER_PORT: $MASTER_PORT"

for DATA_TYPE in "" "text_"
do
    for DATASET_NAME in "syntheticgeometry__triangle" "syntheticgeometry__quadrilateral" "syntheticgeometry__length" "syntheticgeometry__area" "syntheticgeometry__angle" "3d__size" "3d__angle"
    do
        torchrun --master_port=${MASTER_PORT} --nproc_per_node=4 src/training/train_phi35v.py --model_name_or_path microsoft/Phi-3.5-vision-instruct \
            --dataset_jsonl_path dataset/train/phi35v_train_data/${DATA_TYPE}${DATASET_NAME}.jsonl \
            --output_dir finetuning_results/phi35v_finetuning_log \
            --num_train_epochs 3
    done
done

conda deactivate
