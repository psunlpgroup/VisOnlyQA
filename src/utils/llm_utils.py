from typing import Any

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor

from src.utils.hf_utils import check_hf_model_exists
from src.utils.internvl_utils import internvl_split_model
from src.vlmevalkit_utils import is_vlmeval_models


def load_open_model(model_name: str) -> tuple[Any, AutoTokenizer, AutoProcessor]:
    if is_vlmeval_models(model_name):  # VLMEvalKit
        from VLMEvalKit.vlmeval.config import supported_VLM
        return supported_VLM[model_name](), None, None
    
    if "InternVL" in model_name:
        device_map = internvl_split_model(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)

        return model, tokenizer, None
    
    if "Phi-3.5" in model_name:
        # https://huggingface.co/microsoft/Phi-3.5-vision-instruct
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map="cuda", 
            trust_remote_code=True, 
            torch_dtype="auto", 
            _attn_implementation='eager'
            # _attn_implementation='flash_attention_2'  # use this option if you have flash_attn
        )
        
        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            num_crops=4
        )
        
        return model, None, processor

    if "Qwen2-VL" in model_name:
        # https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
        from transformers import Qwen2VLForConditionalGeneration
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name, max_pixels=1280*28*28)
        
        return model, None, processor
    
    if not check_hf_model_exists(model_name):  # proprietary models
        return None, None, None
    
    raise NotImplementedError(f"Model {model_name} is not supported")
