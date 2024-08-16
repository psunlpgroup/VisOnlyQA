import csv
import json
import shutil
import random

from src.typing import VisonlyQA_Instance
from src.path import test_annotations_dir, test_dataset_dir, test_intermediate_dir, test_image_path_dir
from src.config import visonlyqa_response_type_dir
from src.prompts import get_evaluation_prompt, shuffle_options
from src.utils import get_sha512_hash_hex


annotation_csv_files_dict_of_lists = {
    "geometry": [
        "VisOnlyQA_geometry_diagram_annotation - MathVista.csv"
    ],
    "chemistry": [
        "VisOnlyQA_chemistry_annotation - MMMU_Chemistry.csv",
        "VisOnlyQA_chemistry_annotation - MMMU_Chemistry_single.csv"
    ],
    "charts": [
        "VisOnlyQA_charts_annotation - ChartQA.csv",
        "VisOnlyQA_charts_annotation - CharXiv.csv"
    ],
}


def get_split_name(category_name: str, task_category: str) -> str:
    task_category = task_category.lower().replace(", ", "_").replace(" ", "_")
    
    # if category_name == "geometry":
    #     if task_category in ["triangle", "quadrilateral"]:
    #         task_category = "shape"
    
    return f"""{category_name}__{task_category}"""


def get_deta_id(image_path: str, question: str, answer: str) -> str:
    hash = get_sha512_hash_hex(f"{image_path}__{question}__{answer}")
    return f"visonlyqa_{hash}"


if __name__ == "__main__":
    real_test_dataset_dir = test_dataset_dir / "real"
    real_test_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    target_num = 50
    
    annotations: dict[str, list[VisonlyQA_Instance]] = {}
    skipped_split_cache: set[str] = set()  # splits that are not included in visonlyqa_splits
    image_url_dict: dict[str, str] = {}  # save image url for future use (human performance annotation)
    for category_name, annotation_csv_names_list in annotation_csv_files_dict_of_lists.items():
        for annotation_csv_name in annotation_csv_names_list:
            annotation_csv_path = test_annotations_dir / annotation_csv_name
            
            with open(annotation_csv_path, "r") as f:
                csv_reader = csv.reader(f)
                # skip header
                next(csv_reader)
                for row in csv_reader:
                    image_path, image_url, _, q1, a1, q2, a2, task_category = row[:8]
                    split_name = get_split_name(category_name, task_category)
                    
                    if split_name not in visonlyqa_response_type_dir.keys():
                        if split_name not in skipped_split_cache:
                            print(f"{split_name} not implemented. Skipping.")
                            skipped_split_cache.add(split_name)
                        continue
                    
                    image_url_dict[image_path] = image_url
                    
                    for q, a in [[q1, a1], [q2, a2]]:
                        if len(q) == 0:
                            continue
                        
                        question_type, options = visonlyqa_response_type_dir[split_name]
                        q, a = shuffle_options(q, a, options, question_type)
                        
                        annotations.setdefault(split_name, []).append(
                            {
                                "image_path": image_path,
                                "question": q,
                                "answer": a,
                                "prompt_reasoning": get_evaluation_prompt("reasoning", q, response_options=options),
                                "prompt_no_reasoning": get_evaluation_prompt("no_reasoning", q, response_options=options),
                                "image_category": split_name.split("__")[0],
                                "task_category": split_name.split("__")[1],
                                "question_type": question_type,
                                "response_options": options,
                                "source": annotation_csv_name.split(" - ")[1].replace(".csv", ""),
                                "id": get_deta_id(image_path, q, a),
                            }
                        )

                        source_image_path = test_intermediate_dir / image_path
                        destination_image_path = test_dataset_dir / image_path
                        destination_image_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(source_image_path, destination_image_path)
    
    # save dataset
    for split_name, annotation_list in annotations.items():
        annotation_jsonl_path = real_test_dataset_dir / f"{split_name}.jsonl"
        
        annotation_list = random.Random(split_name).sample(annotation_list, len(annotation_list))

        with open(annotation_jsonl_path, "w") as f:
            for annotation in annotation_list[:target_num]:
                json.dump(annotation, f)
                f.write("\n")

    # save image url
    test_image_path_dir.mkdir(parents=True, exist_ok=True)
    image_url_path = test_image_path_dir / "real_image_url.json"
    with open(image_url_path, "w") as f:
        json.dump(image_url_dict, f)
