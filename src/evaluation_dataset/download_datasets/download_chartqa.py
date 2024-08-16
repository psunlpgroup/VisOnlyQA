from pathlib import Path
import json
import shutil
import random
from tqdm import tqdm

from src.path import source_dataset_download_dir, test_intermediate_data_dir, test_intermediate_dir


if __name__ == "__main__":
    dataset_name = "ChartQA"
    dataset_path = source_dataset_download_dir / dataset_name / "ChartQA Dataset/test/test_human.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    # shuffle the dataset
    dataset = random.Random(68).sample(dataset, len(dataset))
    
    # save images
    seedbench_image_dir = Path("images") / dataset_name
    intermediate_seedbench_image_dir = test_intermediate_dir / seedbench_image_dir
    intermediate_seedbench_image_dir.mkdir(parents=True, exist_ok=True)
    
    geometry_diagram_data = []
    for d in tqdm(dataset):
        image_name: str = d["imgname"]
        
        source_image_path: Path = source_dataset_download_dir / dataset_name / "ChartQA Dataset/test/png" / image_name
        target_image_path: Path = intermediate_seedbench_image_dir / image_name
        target_image_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_image_path, target_image_path)
        
        geometry_diagram_data.append(
            {
                "image": str(seedbench_image_dir / image_name),
                "original_question": d["query"],
                "original_answer": d["label"],
            }
        )
    
    # save the charts data
    charts_intermediate_dir = test_intermediate_data_dir / "charts"
    charts_intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    with open(charts_intermediate_dir / f"{dataset_name}.jsonl", "w") as f:
        for data in geometry_diagram_data:
            f.write(json.dumps(data) + "\n")
