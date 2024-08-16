import subprocess

from tap import Tap

from src.config import base_model_to_finetuned_model_dict


class FinetunedEvaluationTap(Tap):
    gpus: str = "0,1,2,3"

if __name__ == "__main__":
    args = FinetunedEvaluationTap().parse_args()
    
    print("In this code, you need to specify GPU ids in args.gpus")
    print(f"You specified GPU ids: {args.gpus}")
    
    for base_model, models_dict in base_model_to_finetuned_model_dict.items():
        for split, model_path in models_dict.items():
            print(f"data={split}, model={model_path}")
            _ = subprocess.run(f"CUDA_VISIBLE_DEVICES={args.gpus} MODEL={model_path} bash shell/4_evaluation/__evaluation_finetuned_models.sh", shell=True)
