# MathVista
python src/evaluation_dataset/download_datasets/download_mathvista.py

# MMMU
python src/evaluation_dataset/download_datasets/download_mmmu_chemistry.py
python src/evaluation_dataset/download_datasets/download_mmmu_chemistry_single.py

mkdir -p ../datasets

# ChartQA
git clone git@github.com:vis-nlp/ChartQA.git ../datasets/ChartQA
python src/evaluation_dataset/download_datasets/download_chartqa.py

# CharXiv
python src/evaluation_dataset/download_datasets/download_charxiv.py

# CLEVR
mkdir -p ../datasets/CLEVR
cd ../datasets/CLEVR
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
cd ../../VisonlyQA
python src/evaluation_dataset/download_datasets/download_clevr.py

# SuperCLEVR
mkdir -p ../datasets/SuperCLEVR
cd ../datasets/SuperCLEVR
wget https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/images.zip
unzip images.zip
rm images.zip
wget https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/superCLEVR_questions_30k.json
wget https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/superCLEVR_scenes.json
python src/evaluation_dataset/download_datasets/download_superclevr.py
