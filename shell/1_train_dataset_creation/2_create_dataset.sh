python src/training_dataset/dataset_creation/create_clevr.py
python src/training_dataset/dataset_creation/create_superclevr.py

# create training data (for internvl) and convert to phi35v and sharegpt format
python src/training_dataset/create_train_data.py
python src/training_dataset/convert_from_internvl_to_phi35v.py
python src/training_dataset/convert_from_internvl_to_sharegpt.py

# set environment variable HF_ACCOUNT for huggingface upload
python src/upload_to_huggingface/upload_to_huggingface_train_set.py

# get dataset statistics
python src/training_dataset/get_dataset_statistics.py
