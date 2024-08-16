from pathlib import Path
import json
from tqdm import tqdm

from datasets import load_dataset
import PIL.Image

from src.path import test_intermediate_data_dir, test_intermediate_dir


if __name__ == "__main__":
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    
    # shuffle the dataset
    dataset = dataset.shuffle(seed=68)
    
    # save images
    mathvista_image_dir = Path("images") / "MathVista"
    intermediate_mathvista_image_dir = test_intermediate_dir / mathvista_image_dir
    intermediate_mathvista_image_dir.mkdir(parents=True, exist_ok=True)
    
    geometry_diagram_data = []
    for d in tqdm(dataset):
        if d["metadata"]["context"] == "geometry diagram":
            image_name = d["image"]
            image_path: Path = intermediate_mathvista_image_dir / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            geometry_diagram_data.append(
                {
                    "image": str(mathvista_image_dir / image_name),
                    "original_question": d["question"] + "\n" + d["query"],
                    "original_answer": d["answer"],
                }
            )
            
            image: PIL.Image.Image = d["decoded_image"]  # PIL image
            image.convert("RGB").save(image_path)
    
    # save the geometry diagram data
    geometry_diagram_intermediate_dir = test_intermediate_data_dir / "geometry_diagram"
    geometry_diagram_intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    with open(geometry_diagram_intermediate_dir / "MathVista.jsonl", "w") as f:
        for data in geometry_diagram_data:
            f.write(json.dumps(data) + "\n")
