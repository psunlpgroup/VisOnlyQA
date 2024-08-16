import json
from pathlib import Path
import copy
import random

from src.path import train_dataset_dir


def convert_convertations_from_internvl_to_phi3v(conversations: list[dict]) -> list[dict]:
    new_conversations: list[dict] = []
    image_idx: int = 1
    
    for utterance in conversations:
        updated_text: str = copy.copy(utterance["value"])
        while "<image>" in updated_text:
            # replace only one "<image>" with "<|image_{image_idx}|>" at each iteration
            updated_text = updated_text.replace("<image>", f"<|image_{image_idx}|>", 1)
            image_idx += 1
        
        new_utterance = {
            "role": {"human": "user", "gpt": "assistant"}[utterance["from"]],
            "content": updated_text,
        }
        new_conversations.append(new_utterance)
    
    return new_conversations


if __name__ == "__main__":
    with open(train_dataset_dir / "internvl_meta/all.json", "r") as f:
        dataset: dict = json.load(f)
    
    phi35v_dir = train_dataset_dir / "phi35v_train_data"
    phi35v_dir.mkdir(parents=True, exist_ok=True)
    
    new_data_list: list[dict] = []
    for split_name, data_info in dataset.items():
        print(split_name)
        
        subdata_list: list[dict] = []
        annotation_jsonl_path = Path(data_info["annotation"][len("../../VisOnlyQA/"):])
        with open(annotation_jsonl_path, "r") as f:
            annotation_list = [json.loads(line) for line in f]
        
        for original_instance in annotation_list:
            new_instance = {
                "image": str(train_dataset_dir / original_instance["image"]),
                "conversations": convert_convertations_from_internvl_to_phi3v(original_instance["conversations"]),
            }
            subdata_list.append(new_instance)
        
        # subset
        with open(phi35v_dir / f"{split_name}.jsonl", "w") as f:
            for instance in subdata_list:
                f.write(json.dumps(instance) + "\n")
        
        new_data_list.extend(subdata_list)
    
    # shuffle
    new_data_list = random.Random(68).sample(new_data_list, len(new_data_list))
    
    output_path = phi35v_dir / "all.jsonl"
    with open(output_path, "w") as f:
        for instance in new_data_list:
            f.write(json.dumps(instance) + "\n")
