python src/evaluation_dataset/postprocess_annotation/postprocess_annotation_test_set.py

# set environment variable HF_ACCOUNT for huggingface upload
python src/upload_to_huggingface/upload_to_huggingface_test_set.py

# get dataset statistics
python src/evaluation_dataset/dataset_statistics/get_dataset_statistics.py
python src/evaluation_dataset/dataset_statistics/get_geometry_statistics.py

# create the human performance annotation csv files
python src/evaluation_dataset/human_performance/generate_human_performance_annotation_interface.py
# after annotation
python src/evaluation_dataset/human_performance/postprocess_human_performance.py
