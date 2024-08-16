conda activate alphageometry

for SEED in {0..100000}
do
    echo "Processing seed $SEED"
    python src/training_dataset/alphageometry/generate_random_geometric_shapes.py --seed $SEED
done

python src/training_dataset/dataset_creation/create_syntheticgeometry.py --split train
python src/training_dataset/dataset_creation/create_syntheticgeometry.py --split val
python src/training_dataset/dataset_creation/create_syntheticgeometry.py --split test

conda deactivate alphageometry
