import os
from typing import Literal

from transformers import AutoConfig

from src.config import (
    open_models_with_specific_code_list,
    huggingface_eval_real_data_name, huggingface_eval_synthetic_data_name, huggingface_eval_synthetic_with_text_data_name, huggingface_train_data_name
)


def check_hf_model_exists(model_name: str) -> bool:
    for pattern in open_models_with_specific_code_list:
        if pattern in model_name:
            return True
    
    try:
        # Try to load the configuration of the model
        _ = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        return True
    except Exception as e:
        return False


def get_hf_dataset_name(split: Literal["train", "eval"]) -> str:
    hf_account = os.getenv("HF_ACCOUNT")
    if hf_account is None:
        raise ValueError("Please set your HuggingFace to HF_ACCOUNT environment variable: export HF_ACCOUNT=your_username")
    
    if split == "train":
        return f"{hf_account}/{huggingface_train_data_name}"
    elif split == "eval_real":
        return f"{hf_account}/{huggingface_eval_real_data_name}"
    elif split == "eval_synthetic":
        return f"{hf_account}/{huggingface_eval_synthetic_data_name}"
    elif split == "eval_synthetic_with_text":
        return f"{hf_account}/{huggingface_eval_synthetic_with_text_data_name}"
    else:
        raise ValueError(f"Invalid split: {split} not in ['train', 'eval_real', 'eval_synthetic']")
