import json
import shutil
from tqdm import tqdm
import random
import copy

from src.typing import VisonlyQA_Instance
from src.path import source_dataset_download_dir, train_dataset_dir, test_dataset_dir, val_dataset_dir
from src.prompts import get_evaluation_prompt, shuffle_options
from src.training_dataset.dataset_creation.create_clevr import get_distance


def get_merged_object_name(name: str) -> str:
    if name in ["airliner", "biplane", "jet"]:
        return "airplane"
    elif name == "fighter":
        return "fighter jet"
    elif name in ["utility", "tandem", "road", "mountain"]:
        return "bike"
    elif name in ["chopper", "scooter", "cruiser", "dirtbike"]:
        return "motorcycle"
    elif name in ["articulated", "double", "regular", "school"]:
        return "bus"
    elif name in ["suv", "minivan", "sedan", "wagon", "truck"]:
        return "car"
    else:
        raise ValueError(f"Unknown object name: {name}")


def filter_objects_superclevr(scene: dict) -> list[tuple[str, dict]]:
    """ Filter out objects that cannot be explained easily (e.g., more than three objects with the same color) """
    
    counter: dict[str, dict[str, list[dict]]] = {}
    for obj in scene["objects"]:
        color = obj["color"]
        shape = get_merged_object_name(obj["shape"])
        
        counter.setdefault(shape, {}).setdefault(color, []).append(obj)
    
    selected_objects: list[tuple[str, dict]] = []
    for obj in scene["objects"]:
        shape = get_merged_object_name(obj["shape"])
        if shape == "bus":
            continue  # we don't use bus because it is often difficult to tell which direction it is facing
        
        objects = counter[shape][obj["color"]]
        
        if len(counter[shape]) == 1:
            if len(objects) == 1:
                # only one object for this color and shape
                selected_objects.append([f"{shape}", obj])
            elif len(objects) == 2:
                if abs(objects[0]["pixel_coords"][0][0] - objects[1]["pixel_coords"][0][0]) < 100:  # too close
                    continue
                
                left = True
                for obj_2 in objects:
                    if obj_2["pixel_coords"][0][0] < obj["pixel_coords"][0][0]:
                        left = False
                        break
                position = "left" if left else "right"
                selected_objects.append([f"{position} {shape}", obj])
            else:
                # too many similar objects
                continue
        else:
            if len(objects) == 1:
                # only one object for this color and shape
                selected_objects.append([f"{obj['color']} {shape}", obj])
            elif len(objects) == 2:
                if abs(objects[0]["pixel_coords"][0][0] - objects[1]["pixel_coords"][0][0]) < 100:  # too close
                    continue
                
                left = True
                for obj_2 in objects:
                    if obj_2["pixel_coords"][0][0] < obj["pixel_coords"][0][0]:
                        left = False
                        break
                position = "left" if left else "right"
                selected_objects.append([f"{position} {obj['color']} {shape}", obj])
            else:
                # too many similar objects
                continue
    
    return selected_objects


def get_rotation(obj: dict) -> float:
    """ Get the rotation of the object. SuperCLEVR uses different rotation for different objects """
    
    shape_name = get_merged_object_name(obj["shape"])
    rotation = obj["rotation"]
    
    if shape_name == "airplane":
        pass  # no need to change
    elif shape_name == "fighter jet":
        pass  # no need to change
    elif shape_name == "bike":
        rotation = (rotation + 270) % 360
    elif shape_name == "motorcycle":
        pass  # no need to change
    elif shape_name == "bus":
        pass
        # raise ValueError("Bus should not be used")
    elif shape_name == "car":
        pass  # no need to change
    else:
        raise ValueError(f"Unknown object name: {shape_name}")
    
    return rotation


def get_all_angles(selected_objects: list[tuple[str, dict]]) -> list[tuple[str, tuple[str, str], tuple[dict, dict]]]:
    angles = []
    for idx_1 in range(len(selected_objects)):
        for idx_2 in range(idx_1 + 1, len(selected_objects)):
            name_1, obj_1 = selected_objects[idx_1]
            name_2, obj_2 = selected_objects[idx_2]
            
            angle = abs(get_rotation(obj_1) - get_rotation(obj_2))
            if angle > 180:
                angle = 360 - angle
            
            if angle < 5:
                option = "a"
            elif 40 < angle and angle < 50:
                option = "b"
            elif 85 < angle and angle < 95:
                option = "c"
            elif 130 < angle and angle < 140:
                option = "d"
            elif 175 < angle:
                option = "e"
            else:
                continue
            
            angles.append((option, (name_1, name_2), (obj_1, obj_2)))
    
    return angles


def convert_visual_information_to_text_superclevr(scene: dict) -> str:
    output: str = "The following text provides information about 3D objects in the provided figure.\n\n"
    for idx, obj in enumerate(scene["objects"]):
        shape_name = get_merged_object_name(obj["shape"])
        color = obj["color"]
        rotation = get_rotation(obj)
        position = f"""({obj["pixel_coords"][0][0]:d}, {obj["pixel_coords"][0][1]:d})"""
        
        output += f"Object {idx+1}: Shape={shape_name}, Color={color}, Rotation={rotation:.0f}, Position={position}\n"
    
    output += "\nUsing the figure and above information, answer the following questions.\n\n"
    
    return output


angle_prompt_template = "Which of the following options is a reasonable estimate of the angle between the directions of the {object_1} and the {object_2} in the figure? We define 0 degrees if the direction is the same and 180 degrees if the direction is opposite. (a) 0 degrees (b) 45 degrees (c) 90 degrees (d) 135 degrees (e) 180 degrees"


if __name__ == "__main__":
    dataset_name = "SuperCLEVR"
    
    print("loading original dataset")
    data_path = source_dataset_download_dir / dataset_name / "superCLEVR_scenes.json"
    with open(data_path, "r") as f:
        scenes_list: list[dict] = json.load(f)["scenes"]
    
    # shuffle the dataset
    scenes_list = random.Random(68).sample(scenes_list, len(scenes_list))
    scenes_list_dict = {
        "test": scenes_list[:500],
        "val": scenes_list[500:1000],
        "train": scenes_list[1000:],
    }
    
    for split in ["train", "val", "test"]:
        print(f"Creating {split} data")
        
        total_dataset_num = 10000 if split == "train" else 100
        
        dataset: list[VisonlyQA_Instance] = []
        text_dataset: list[VisonlyQA_Instance] = []
        for scene in tqdm(scenes_list_dict[split]):
            filtered_objects = filter_objects_superclevr(scene)
            if len(filtered_objects) < 5:  # we use images with many objects
                continue
            
            angles_list = get_all_angles(filtered_objects)
            if len(angles_list) == 0:
                continue
            
            data_identifier = scene["image_filename"].replace(".png", "")
            source_path = source_dataset_download_dir / dataset_name / "images" / f"{data_identifier}.png"
            image_path = f"images/SuperCLEVR/{data_identifier}.png"
            dataset_dir = {"train": train_dataset_dir, "val": val_dataset_dir, "test": test_dataset_dir}[split]
            target_path = dataset_dir / image_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, target_path)
            
            selected_objects_list = []
            for idx in range(5):
                selected_objcets = random.Random(f"{data_identifier}_{str(idx)}").choice(angles_list)
                o1, o2 = selected_objcets[2]
                distance = get_distance(o1, o2)
                selected_objects_list.append([distance, selected_objcets])
            
            answer, (obj1, obj2), (_, _) = max(selected_objects_list, key=lambda x: x[0])[1] # get the farthest objects
            
            question = angle_prompt_template.format(object_1=obj1, object_2=obj2)
            options_list = ["a", "b", "c", "d", "e"]
            question, answer = shuffle_options(question, answer, options=options_list, question_type="single_answer")
            
            dataset.append(
                {
                    "image_path": image_path,
                    "question": None if split in ["train", "val"] else question,
                    "answer": answer,
                    "prompt_reasoning": None if split  in ["train", "val"] else get_evaluation_prompt("reasoning", question, response_options=options_list),
                    "prompt_no_reasoning": get_evaluation_prompt("no_reasoning", question, response_options=options_list),
                    "image_category": "3d",
                    "task_category": "size",
                    "question_type": "single_answer",
                    "response_options": options_list,
                    "source": dataset_name,
                    "id": data_identifier,
                }
            )
            
            # text dataset, which provides visual information in the texual format as well
            text_d = copy.deepcopy(dataset[-1])
            
            visual_information_in_text = convert_visual_information_to_text_superclevr(scene)
            for key in ["question", "prompt_reasoning", "prompt_no_reasoning"]:
                if text_d[key] is not None:
                    text_d[key] = visual_information_in_text + "\n\n" + text_d[key]
            text_d["id"] = "text_" + text_d["id"]
            
            text_dataset.append(text_d)
            
            if len(dataset) >= total_dataset_num:
                break
        
        # save
        for data_list in [dataset, text_dataset]:
            save_dir = {"train": train_dataset_dir, "val": val_dataset_dir, "test": test_dataset_dir}[split]
            filename = "3d__angle.jsonl"

            if "text_" in data_list[0]["id"]:
                save_dir = save_dir / "synthetic_with_text"
                filename = "text_" + filename
            else:
                save_dir = save_dir / "synthetic"
            
            save_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = save_dir / filename
            
            with open(output_path, "w") as f:
                for data in data_list:
                    f.write(json.dumps(data) + "\n")
