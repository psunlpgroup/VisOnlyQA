from pathlib import Path
import json
from tqdm import tqdm

import PIL.Image
from datasets import load_dataset

from src.path import test_intermediate_data_dir, test_intermediate_dir


if __name__ == "__main__":
    dataset_name = "CharXiv"
    figure_category = "charts"
    
    dataset = load_dataset("princeton-nlp/CharXiv", split="validation")
    
    # shuffle the dataset
    dataset = dataset.shuffle(seed=68)
    
    # save images
    image_save_dir = Path("images") / dataset_name
    intermediate_image_dir = test_intermediate_dir / image_save_dir
    intermediate_image_dir.mkdir(parents=True, exist_ok=True)
    
    text_data = []
    for d in tqdm(dataset):
        image_name_original: str = d["original_figure_path"]
        image_name = image_name_original.replace("/", "--")
        image_path: Path = intermediate_image_dir / image_name
        image_path.parent.mkdir(parents=True, exist_ok=True)
        text_data.append(
            {
                "image": str(image_save_dir / image_name),
                "original_question": d["reasoning_q"],
                "original_answer": d["reasoning_a"],
            }
        )
        
        image: PIL.Image.Image = d["image"]
        image.convert("RGB").save(image_path)
    
    # save
    intermediate_output_dir = test_intermediate_data_dir / figure_category
    intermediate_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(intermediate_output_dir / f"{dataset_name}.jsonl", "w") as f:
        for data in text_data:
            f.write(json.dumps(data) + "\n")
