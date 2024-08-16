from pathlib import Path
import json
import hashlib
from typing import Optional

import torch
import PIL.Image
from transformers import AutoTokenizer, AutoModel, AutoProcessor

from src.config import proprietary_models_list
from src.evaluation.utils import get_model_input, encode_image_base64
from src.utils import check_hf_model_exists, internvl_load_image
from src.vlmevalkit_utils import is_vlmeval_models
from src.path import test_dataset_dir


class FileEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, PIL.Image.Image):
            return encode_image_base64(obj)
        return super().default(obj)


def dict_to_cache_path(input_dict: dict, cache_dir = Path("./cache")):
    str_dict = json.dumps(input_dict, sort_keys=True, cls=FileEncoder)
    
    hash_object = hashlib.sha512(str_dict.encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    return cache_dir / f"{hex_dig}.json"


def add_image_hash_to_input_dict(input_dict: dict, image: Optional[PIL.Image.Image]):
    if image is None:
        return input_dict
    
    image_hash = hashlib.sha256(image.tobytes()).hexdigest()
    return {**input_dict, "image_hash": image_hash}


def read_cache(input_dict: dict, image: Optional[PIL.Image.Image], cache_dir = Path("./cache")):
    input_dict = add_image_hash_to_input_dict(input_dict, image)
    cache_path = dict_to_cache_path(input_dict, cache_dir)
    
    if not Path(cache_path).exists():
        return None
    
    with open(cache_path, "r") as f:
        return json.load(f)


def dump_cache(cache: dict, input_dict: dict, image: PIL.Image.Image, cache_dir = Path("./cache")):
    input_dict = add_image_hash_to_input_dict(input_dict, image)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dict_to_cache_path(input_dict, cache_dir)
    
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=4)


def get_model_specific_input_dict(model: str, seed: int=None) -> dict:
    if "gpt" in model:
        return {"temperature": 0.0, "seed": seed}
    
    if "claude" in model:
        return {"temperature": 0.0}
    
    if "gemini" in model:
        return {}
    
    # is_vlmeval_models should come before check_hf_model_exists because the models in VLMEval are mostly also in huggingface
    if is_vlmeval_models(model):
        return {}
    
    if check_hf_model_exists(model):
        return {"do_sample": False}

    raise NotImplementedError(f"Model {model} is not supported")


def call_llm(model_name: str, prompt: str, image: PIL.Image.Image=None, image_path: str=None,
             open_model: Optional[AutoModel]=None, hf_tokenizer: Optional[AutoTokenizer]=None, hf_processor: Optional[AutoProcessor]=None,
             max_tokens=1048, seed=68) -> str:
    message = get_model_input(model_name, prompt, image)
    input_dict = {
        "model": model_name,
        "messages": message,
        "max_tokens": max_tokens,
    }
    input_dict.update(get_model_specific_input_dict(model=model_name, seed=seed))
    
    cache = read_cache(input_dict, image=None if model_name in proprietary_models_list else image)
    save_cache = True if cache is None else False
    
    if "gpt" in model_name:
        if cache is None:
            import openai
            
            with open("../openai_api_key.txt", "r") as f:
                api_key = f.read().strip()
            
            client = openai.OpenAI(api_key=api_key)
            cache = client.chat.completions.create(**input_dict).dict()
        
        response = cache["choices"][0]["message"]["content"]
    elif "claude" in model_name:
        import anthropic
        
        if cache is None:
            with open("../anthropic_api_key.txt", "r") as f:
                api_key = f.read().strip()
            
            client = anthropic.Anthropic(api_key=api_key)
            cache = client.messages.create(**input_dict).dict()
        
        response = cache["content"][0]["text"]
    elif "gemini" in model_name:
        if cache is None:
            import google.generativeai as genai
            with open("../google_api_key.txt", "r") as f:
                api_key = f.read().strip()
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel(model_name)
            cache = model.generate_content(input_dict["messages"]).text
        
        response = cache

    elif "InternVL" in model_name:
        if cache is None:
            pixel_values = internvl_load_image(image, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=max_tokens, do_sample=False)
            cache = open_model.chat(hf_tokenizer, pixel_values, message, generation_config)
        
        response = cache
    elif "Phi-3.5" in model_name:
        if cache is None:
            messages = [
                {"role": "user", "content": prompt},
            ]
            prompt = hf_processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = hf_processor(prompt, [image], return_tensors="pt").to("cuda:0") 
            generation_args = { 
                "max_new_tokens": max_tokens,
                "temperature": 0.0, 
                "do_sample": False, 
            }
            generate_ids = open_model.generate(**inputs, 
                eos_token_id=hf_processor.tokenizer.eos_token_id, 
                **generation_args
            )
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            cache = hf_processor.batch_decode(
                generate_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
        response = cache
    elif "Qwen2-VL" in model_name:
        if cache is None:
            # https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
            from qwen_vl_utils import process_vision_info
            
            messages = input_dict["messages"]
            text = hf_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = hf_processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = open_model.generate(**inputs, max_new_tokens=2048)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            cache = hf_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        response = cache
    elif is_vlmeval_models(model_name):
        if cache is None:
            full_image_path = str(test_dataset_dir / image_path)
            cache = open_model.generate([full_image_path, prompt], dataset="")
        
        response = cache
    elif check_hf_model_exists(model_name):
        # TODO
        raise NotImplementedError(f"Model {model_name} is not supported")
    else:
        raise NotImplementedError(f"Model {model_name} is not supported")
    
    if save_cache:
        dump_cache(cache, input_dict, image=None if model_name in proprietary_models_list else image)
    return response
