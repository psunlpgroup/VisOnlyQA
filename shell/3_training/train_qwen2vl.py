import subprocess

from tap import Tap

from src.config import train_data_splits, train_data_text_splits


class Qwen2vlTrainTap(Tap):
    gpus: str = "0,1,2,3"

if __name__ == "__main__":
    args = Qwen2vlTrainTap().parse_args()
    
    for model_size in [2, 7]:
        for split in train_data_splits + train_data_text_splits:
            print(f"model={model_size}B, data={split}")
            
            _ = subprocess.run(f"GPUS={args.gpus} MODEL_SIZE={model_size} SPLIT={split} bash shell/3_training/__train_qwen2vl.sh", shell=True)
