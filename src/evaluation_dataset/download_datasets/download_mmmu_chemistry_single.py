from pathlib import Path
import json
from tqdm import tqdm
import ast

from datasets import load_dataset
import PIL.Image

from src.path import test_intermediate_data_dir, test_intermediate_dir


if __name__ == "__main__":
    dataset_name = "MMMU_Chemistry_single"
    
    dataset = load_dataset("MMMU/MMMU", name="Chemistry", split="test")
    
    # shuffle the dataset
    dataset = dataset.shuffle(seed=68)
    
    # save images
    image_save_dir = Path("images") / dataset_name
    intermediate_image_dir = test_intermediate_dir / image_save_dir
    intermediate_image_dir.mkdir(parents=True, exist_ok=True)
    
    text_data = []
    for d in tqdm(dataset):
        if d["img_type"] != "['Chemical Structures']":
            continue
        
        # only one figure
        if d["image_2"] is not None:
            continue
        
        image_name = d["id"] + ".png"
        image_path: Path = intermediate_image_dir / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        text_data.append(
            {
                "image": str(image_save_dir / image_name),
                # "original_question": d["question"] + "\n" + d["query"],
                # "original_answer": d["answer"],
            }
        )
        
        image: PIL.Image.Image = d["image_1"]
        image.convert("RGB").save(image_path)
    
    # save
    intermediate_output_dir = test_intermediate_data_dir / "chemistry"
    intermediate_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(intermediate_output_dir / f"{dataset_name}.jsonl", "w") as f:
        for data in text_data:
            f.write(json.dumps(data) + "\n")
