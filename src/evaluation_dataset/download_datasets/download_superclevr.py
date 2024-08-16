from pathlib import Path
import json
import shutil
from tqdm import tqdm
import random

from src.path import source_dataset_download_dir, test_intermediate_data_dir, test_intermediate_dir


if __name__ == "__main__":
    dataset_name = "SuperCLEVR"
    
    data_path = source_dataset_download_dir / dataset_name / "superCLEVR_questions_30k.json"
    with open(data_path, "r") as f:
        dataset: list[dict] = json.load(f)["questions"]
    
    # shuffle the dataset
    dataset = random.Random(68).sample(dataset, len(dataset))[:300]
    
    # save images
    clevr_image_dir = Path("images") / dataset_name
    intermediate_clevr_image_dir = test_intermediate_dir / clevr_image_dir
    intermediate_clevr_image_dir.mkdir(parents=True, exist_ok=True)
    
    data_list = []
    for d in tqdm(dataset):
        image_name = d["image_filename"]
        
        source_image_path: Path = source_dataset_download_dir / dataset_name / "images" / image_name
        target_image_path: Path = intermediate_clevr_image_dir / image_name
        target_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_image_path, target_image_path)
        
        data_list.append(
            {
                "image": str(clevr_image_dir / image_name),
                "original_question": d["question"],
                "original_answer": d["answer"],
            }
        )

    # save the synthetic image data
    synthetic_images_intermediate_dir = test_intermediate_data_dir / "synthetic_images"
    synthetic_images_intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    with open(synthetic_images_intermediate_dir / f"{dataset_name}.jsonl", "w") as f:
        for data in data_list:
            f.write(json.dumps(data) + "\n")
