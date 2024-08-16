from pathlib import Path
import json
from tqdm import tqdm
import ast

from datasets import load_dataset
import PIL.Image

from src.path import test_intermediate_data_dir, test_intermediate_dir
from src.utils import concatenate_pil_images_horizontally


if __name__ == "__main__":
    dataset_name = "MMMU_Chemistry"
    
    dataset = load_dataset("MMMU/MMMU", name="Chemistry", split="test")
    
    # shuffle the dataset
    dataset = dataset.shuffle(seed=68)
    
    # save images
    image_save_dir = Path("images") / dataset_name
    intermediate_image_dir = test_intermediate_dir / image_save_dir
    intermediate_image_dir.mkdir(parents=True, exist_ok=True)
    
    text_data = []
    for d in tqdm(dataset):
        options = ast.literal_eval(d["options"])
        if len(options) < 4:
            continue
        
        if options[0] == "<image 1>":
            if options[3] != "<image 4>":
                continue
            images_list: list[PIL.Image.Image] = [d["image_1"], d["image_2"], d["image_3"], d["image_4"]]
        elif options[0] == "<image 2>":
            if options[3] != "<image 5>":
                continue
            images_list: list[PIL.Image.Image] = [d["image_2"], d["image_3"], d["image_4"], d["image_5"]]
        else:
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
        
        image = concatenate_pil_images_horizontally(images_list)
        image.convert("RGB").save(image_path)
    
    # save
    intermediate_output_dir = test_intermediate_data_dir / "chemistry"
    intermediate_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(intermediate_output_dir / f"{dataset_name}.jsonl", "w") as f:
        for data in text_data:
            f.write(json.dumps(data) + "\n")
