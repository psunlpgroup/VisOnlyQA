# Training data in a format of sharegpt in https://github.com/hiyouga/LLaMA-Factory
# We use this for fine-tuning Qwen2-VL

import json
from pathlib import Path
import copy

from src.path import train_dataset_dir, llama_factory_data_info_path, config_dir


# https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_full/qwen2vl_full_sft.yaml
# https://github.com/QwenLM/Qwen2-VL?tab=readme-ov-file#training

# We assume to run on 4 GPUs
# Total batch size: 128
qwen_yaml = """### model
model_name_or_path: Qwen/Qwen2-VL-{model_size}B-Instruct

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: visonlyqa_{split}
template: qwen2_vl
cutoff_len: 2048
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: ../VisOnlyQA/finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-{model_size}B-Instruct_finetuned_{split}
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_steps: 100
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.005
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 10
"""


def convert_convertations_from_internvl_to_sharegpt(conversations: list[dict]) -> list[dict]:
    new_conversations: list[dict] = []
    
    for utterance in conversations:
        updated_text: str = copy.copy(utterance["value"])
        new_utterance = {
            "role": {"human": "user", "gpt": "assistant"}[utterance["from"]],
            "content": updated_text,
        }
        new_conversations.append(new_utterance)
    
    return new_conversations


if __name__ == "__main__":
    with open(train_dataset_dir / "internvl_meta/all.json", "r") as f:
        dataset: dict = json.load(f)
    
    sharegpt_dir = train_dataset_dir / "sharegpt_train_data"
    sharegpt_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, data_info in dataset.items():
        subdata_list: list[dict] = []
        annotation_jsonl_path = Path(data_info["annotation"][len("../../VisOnlyQA/"):])
        with open(annotation_jsonl_path, "r") as f:
            annotation_list = [json.loads(line) for line in f]
        
        for original_instance in annotation_list:
            new_instance = {
                "images": [str(Path("../../VisOnlyQA") / train_dataset_dir / original_instance["image"])],
                "messages": convert_convertations_from_internvl_to_sharegpt(original_instance["conversations"]),
            }
            subdata_list.append(new_instance)
        
        # subset
        with open(sharegpt_dir / f"{split_name}.json", "w") as f:
            f.write("[\n")
            for i, instance in enumerate(subdata_list):
                f.write(json.dumps(instance))
                if i < len(subdata_list) - 1:
                    f.write(",\n")
            f.write("\n]")
    
    # update data info of LLaMA Factory
    with open(llama_factory_data_info_path, "r") as f:
        data_info = json.load(f)
    
    for split in dataset.keys():
        dataset_path = Path("../../VisOnlyQA") / sharegpt_dir / f"{split}.json"
        
        data_info[f"visonlyqa_{split}"] = {
            "file_name": str(dataset_path),
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
                "images": "images"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }
        }
    
    with open(llama_factory_data_info_path, "w") as f:
        json.dump(data_info, f, indent=2)
    
    # create yaml
    yaml_output_dir = config_dir / "qwen2vl_train_yaml"
    yaml_output_dir.mkdir(parents=True, exist_ok=True)
    for model_size in [2, 7]:
        output_dir = yaml_output_dir / f"Qwen2-VL-{model_size}B-Instruct"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in dataset.keys():
            yaml_content = qwen_yaml.format(model_size=model_size, split=split)
            
            output_path = output_dir / f"{split}.yaml"
            with open(output_path, "w") as f:
                f.write(yaml_content)
