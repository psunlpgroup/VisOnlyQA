import json
from pathlib import Path
import random

from src.path import train_dataset_dir
from src.config import train_data_splits, train_data_text_splits


if __name__ == "__main__":
    print("Loading data")
    all_data_list_dict: dict[str, list] = {}
    for dataset_name in train_data_splits + train_data_text_splits:
        data_path = train_dataset_dir / "synthetic" / f"{dataset_name}.jsonl"
        with open(data_path, "r") as f:
            data_list = [json.loads(line) for line in f]
        all_data_list_dict[dataset_name] = data_list
    
    subdata_dir = train_dataset_dir / "first_50_data"
    subdata_dir.mkdir(parents=True, exist_ok=True)
    
    # merge data
    all_data_list = []
    for data_name, data_list in all_data_list_dict.items():
        all_data_list.extend(data_list)
        
        # data for evaluating performance on training data
        with open(subdata_dir / f"{data_name}_50.jsonl", "w") as f:
            for data in random.Random(68).sample(data_list, 50):
                f.write(json.dumps(data) + "\n")
    
    all_data_list = random.Random(68).sample(all_data_list, len(all_data_list))
    
    with open(train_dataset_dir / "train_all.jsonl", "w") as f:
        for data in all_data_list:
            f.write(json.dumps(data) + "\n")
    
    with open(train_dataset_dir / "train_all_first_100.jsonl", "w") as f:
        for data in all_data_list[:100]:
            f.write(json.dumps(data) + "\n")
    
    # InternVL
    print("Creating InternVL data")
    internvl_meta = {}
    for dataset_name, data_list in all_data_list_dict.items():
        annotations_path = train_dataset_dir / "internvl_annotations" / f"{dataset_name}.jsonl"
        annotations_path.parent.mkdir(parents=True, exist_ok=True)
        
        annotations = []
        for idx, d in enumerate(data_list):
            annotations.append(
                {
                    "id": idx, "image": d["image_path"],
                    "conversations": [{"from": "human", "value": f"<image>\n{d['prompt_no_reasoning']}"}, {"from": "gpt", "value": d["answer"]}]
                }
            )
        
        with open(annotations_path, "w") as f:
            for annotation in annotations:
                json.dump(annotation, f)
                f.write("\n")
        
        # meta information for internvl
        visonlyqa_dir = Path("../../VisOnlyQA")  # relative to the internvl directory
        internvl_meta[dataset_name] = {
            "root": str(visonlyqa_dir / train_dataset_dir),
            "annotation": str(visonlyqa_dir / annotations_path),
            "data_augment": False,
            "repeat_time": 1,
            "length": len(data_list),
        }
    
    internvl_train_data_dir = train_dataset_dir / "internvl_meta"
    internvl_train_data_dir.mkdir(parents=True, exist_ok=True)
    with open(internvl_train_data_dir / f"all.json", "w") as f:
        json.dump(internvl_meta, f, indent=4)
    
    for split, data_list in all_data_list_dict.items():
        with open(internvl_train_data_dir / f"{split}.json", "w") as f:
            json.dump({split: internvl_meta[split]}, f, indent=4)
