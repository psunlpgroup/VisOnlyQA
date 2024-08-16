import json
import shutil
import random
import copy
from pathlib import Path

from tqdm import tqdm

from src.typing import VisonlyQA_Instance
from src.path import (
    source_dataset_download_dir,
    train_dataset_dir, test_dataset_dir, val_dataset_dir
)
from src.prompts import get_evaluation_prompt, shuffle_options


clevr_dir = source_dataset_download_dir / "CLEVR" / "CLEVR_v1.0"


def get_distance(obj1: dict, obj2: dict) -> float:
    """ Get the distance between two objects """
    
    x1, y1 = obj1["3d_coords"][0], obj1["3d_coords"][1]
    x2, y2 = obj2["3d_coords"][0], obj2["3d_coords"][1]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def filter_objects(scene: dict) -> list[tuple[str, dict]]:
    """ Filter out objects that cannot be explained easily (e.g., more than three objects with the same color) """
    
    counter: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for obj in scene["objects"]:
        color = obj["color"]
        shape = obj["shape"]
        material = obj["material"]
        
        counter.setdefault(shape, {}).setdefault(color, {}).setdefault(material, []).append(obj)
    
    selected_objects: list[tuple[str, dict]] = []
    for obj in scene["objects"]:
        objects = counter[obj["shape"]][obj["color"]][obj["material"]]
        
        if len(counter[obj["shape"]][obj["color"]]) == 1:  # only one material
            if len(objects) == 1:  # only one object for this color and shape
                selected_objects.append([f"{obj['color']} {obj['shape']}", obj])
            elif len(objects) == 2:
                if abs(objects[0]["pixel_coords"][0] - objects[1]["pixel_coords"][0]) < 100:  # too close
                    continue
                
                left = True
                for obj_2 in objects:
                    if obj_2["pixel_coords"][0] < obj["pixel_coords"][0]:
                        left = False
                        break
                position = "left" if left else "right"
                selected_objects.append([f"{position} {obj['color']} {obj['shape']}", obj])
            else:  # too many objects
                continue
        else:  # multiple materials
            if len(objects) == 1:  # only one object for this color, shape, and material
                selected_objects.append([f"{obj['color']} {obj['material']} {obj['shape']}", obj])
            elif len(objects) == 2:
                if abs(objects[0]["pixel_coords"][0] - objects[1]["pixel_coords"][0]) < 100:  # too close
                    continue
                
                left = True
                for obj_2 in objects:
                    if obj_2["pixel_coords"][0] < obj["pixel_coords"][0]:
                        left = False
                        break
                position = "left" if left else "right"
                selected_objects.append([f"{position} {obj['color']} {obj['material']} {obj['shape']}", obj])
            else:  # too many objects
                continue
    
    return selected_objects


def get_size_comparison_answer(obj1: dict, obj2: dict) -> str:
    """ Get the answer for the size comparison question """
    
    obj1_size = obj1["size"]
    obj2_size = obj2["size"]
    
    if obj1_size == obj2_size:
        return "b"  # same
    elif obj1_size == "large":
        return "c"  # obj1 is larger
    elif obj2_size == "large":
        return "a"  # obj2 is larger
    else:
        raise Exception("Invalid size")


size_prompt_template = """The {obj1} is X times {tallerwider} than the {obj2}. Which of the following options is a reasonable estimate? (a) 0.5 (b) 1 (c) 2"""


def convert_visual_information_to_text_clevr(scene: dict) -> str:
    output: str = "The following text provides information on the color and size of 3D objects in the provided figure.\n\n"
    for idx, obj in enumerate(scene["objects"]):
        color = obj["color"]
        shape = obj["shape"]
        material = obj["material"]
        
        size = 2 if obj["size"] == "large" else 1
        position = f"""({obj["pixel_coords"][0]:d}, {obj["pixel_coords"][1]:d})"""
        
        output += f"Object {idx+1}: Color={color}, Shape={shape}, Material={material}, Width={size}, Height={size}, Position={position}\n"
    
    output += "\nUsing the figure and above information, answer the following questions.\n\n"
    
    return output


def get_save_path(split: str, is_text_dataset: bool=False) -> Path:
    save_dir = {"train": train_dataset_dir, "val": val_dataset_dir, "test": test_dataset_dir}[split]
    
    if is_text_dataset:
        save_dir = save_dir / "synthetic_with_text"
    else:
        save_dir = save_dir / "synthetic"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if is_text_dataset:
        return save_dir / "text_3d__size.jsonl"
    else:
        return save_dir / "3d__size.jsonl"


if __name__ == "__main__":
    dataset_name = "CLEVR"
    
    for split in ["train", "val"]:
        if split == "train":
            print("Creating training data")
        else:
            print("Creating test and validation data")
        
        total_dataset_num = 10000 if split == "train" else 100 * 2  # 100 for test, 100 for validation
        
        with open(clevr_dir / "scenes" / f"CLEVR_{split}_scenes.json", "r") as f:
            scenes_list = json.load(f)["scenes"]
        scenes_list = random.Random(68).sample(scenes_list, len(scenes_list))
        
        dataset_dir = train_dataset_dir if split == "train" else test_dataset_dir
        images_dir = dataset_dir / "images" / dataset_name
        images_dir.mkdir(parents=True, exist_ok=True)
        
        saved_split = "train" if split == "train" else "test"

        dataset: list[VisonlyQA_Instance] = []
        text_dataset: list[VisonlyQA_Instance] = []
        for scene in tqdm(scenes_list):
            if split == "val" and len(dataset) == total_dataset_num // 2:  # the latter half is for the validation set
                dataset_dir = val_dataset_dir
                images_dir = dataset_dir / "images" / dataset_name
                images_dir.mkdir(parents=True, exist_ok=True)
                
                saved_split = "val"
            
            data_idx = scene["image_index"]
            
            data_identifier = f"CLEVR_{split}_{data_idx:06}"
            original_image_path = clevr_dir / "images" / split / f"{data_identifier}.png"
            image_path = f"images/CLEVR/{data_identifier}.png"
            shutil.copy(original_image_path, dataset_dir / image_path)
            
            filtered_objects = filter_objects(scene)
            if len(filtered_objects) < 5:  # we use images with many objects
                continue
            
            # create questions
            object_list = []
            for idx in range(5):
                o1, o2 = random.Random(data_idx + idx).sample(filtered_objects, 2)
                distance = get_distance(o1[1], o2[1])
                object_list.append([o1, o2, distance])
            
            obj1, obj2, _ = max(object_list, key=lambda x: x[2])  # get the farthest objects
            answer = get_size_comparison_answer(obj1[1], obj2[1])
            
            question = size_prompt_template.format(obj1=obj1[0], obj2=obj2[0], tallerwider=random.Random(data_idx).choice(["taller", "wider"]))
            options = ["a", "b", "c"]
            question, answer = shuffle_options(question, answer, options=options, question_type="single_answer")
            
            dataset.append(
                {
                    "image_path": image_path,
                    "question": None if saved_split in ["train", "val"] else question,
                    "answer": answer,
                    "prompt_reasoning": None if saved_split in ["train", "val"] else get_evaluation_prompt("reasoning", question, response_options=options),
                    "prompt_no_reasoning": get_evaluation_prompt("no_reasoning", question, response_options=options),
                    "image_category": "3d",
                    "task_category": "size",
                    "question_type": "single_answer",
                    "response_options": options,
                    "source": dataset_name,
                    "id": data_identifier,
                }
            )
            
            # text dataset, which provides visual information in the texual format as well
            text_d = copy.deepcopy(dataset[-1])
            
            visual_information_in_text = convert_visual_information_to_text_clevr(scene)
            for key in ["question", "prompt_reasoning", "prompt_no_reasoning"]:
                if text_d[key] is not None:
                    text_d[key] = visual_information_in_text + "\n\n" + text_d[key]
            text_d["id"] = "text_" + text_d["id"]
            
            text_dataset.append(text_d)
            
            if len(dataset) >= total_dataset_num:
                break
        
        save_data_list: list[str, list[dict]] = []
        text_save_data_list: list[str, list[dict]] = []
        if split == "train":
            output_path = get_save_path("train")
            save_data_list.append([output_path, dataset])
            
            text_output_path = get_save_path("train", is_text_dataset=True)
            text_save_data_list.append([text_output_path, text_dataset])
        else:
            test_data, val_data = dataset[:100], dataset[100:]
            
            save_data_list.append([get_save_path("test"), test_data])
            save_data_list.append([get_save_path("val"), val_data])
            
            text_test_data, text_val_data = text_dataset[:100], text_dataset[100:]
            
            text_save_data_list.append([get_save_path("test", is_text_dataset=True), text_test_data])
            text_save_data_list.append([get_save_path("val", is_text_dataset=True), text_val_data])
        
        for path, saved_dataset in save_data_list + text_save_data_list:
            with open(path, "w") as f:
                for data in saved_dataset:
                    f.write(json.dumps(data) + "\n")
