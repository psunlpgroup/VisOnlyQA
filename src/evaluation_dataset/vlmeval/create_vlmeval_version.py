import csv
import random

import datasets
from huggingface_hub import HfApi

from src.typing import VisonlyQA_Instance
from src.utils import get_hf_dataset_name
from src.config import visonlyqa_real_splits, visonlyqa_synthetic_splits, visonlyqa_synthetic_with_text_splits
from src.path import intermediate_dir
from src.prompts import extract_options_from_question

from vlmeval.smp.vlm import encode_image_to_base64


visonlyqa_tsv_path = intermediate_dir / "visonlyqa_tsv" / "visonlyqa_vlmevalkit.tsv"

option_names_list = ["A", "B", "C", "D", "E"]


def get_options_for_vlmeval(d: VisonlyQA_Instance) -> tuple[str, list[str], str]:
    output = [None for _ in option_names_list]
    
    if "True" in d["response_options"]:
        options_dict = {"A": "The statement is true", "B": "The statement is false"}
        question = d["question"]
        
        if d["answer"] == "True":
            answer = "A"
        else:
            assert d["answer"] == "False"
            answer = "B"
    else:
        assert "a" in d["response_options"]
        extracted_dict = extract_options_from_question(d["question"], d["response_options"])
        
        question = extracted_dict["question"]
        options_dict: dict = {key.upper(): value for key, value in extracted_dict["options"].items()}
        answer = d["answer"].upper()
    
    # make an option list
    for key, value in options_dict.items():
        output[option_names_list.index(key)] = value
    
    return question, output, answer


if __name__ == "__main__":
    splits_dir = {"real": visonlyqa_real_splits, "synthetic": visonlyqa_synthetic_splits}
    
    list_of_list: list[list[str]] = []
    for real_synthetic in ["real", "synthetic"]:
        repository_name = get_hf_dataset_name(f"eval_{real_synthetic}")
        
        for split in splits_dir[real_synthetic]:
            if split == "chemistry__shape_multi":
                print("Skipping chemistry__shape_multi")
                continue
            
            dataset = datasets.load_dataset(repository_name, split=split)
            print(f"Dataset {real_synthetic} {split} has {len(dataset)} instances")
            
            for d in dataset:
                image = encode_image_to_base64(d["decoded_image"])
                question, options, answer = get_options_for_vlmeval(d)
                
                # index will be added later
                list_of_list.append([None, image, d["image_path"], question, answer, f"Eval_{real_synthetic.capitalize()}", split, d["id"]] + options)
    
    print("Total number of instances:", len(list_of_list))
    
    # shuffle the list
    list_of_list = random.Random(68).sample(list_of_list, len(list_of_list))
    
    # add index
    for index in range(len(list_of_list)):
        list_of_list[index][0] = str(index)
    
    # add header
    list_of_list = [["index", "image", "image_path", "question", "answer", "split", "category", "unique_id"] + option_names_list] + list_of_list
    
    # write to tsv
    visonlyqa_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(visonlyqa_tsv_path, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(list_of_list)
    
    ###
    # upload vlmevalkit version
    repository_name = get_hf_dataset_name(f"eval_real")
    api = HfApi()
    api.upload_file(repo_type="dataset", repo_id=repository_name, path_in_repo="visonlyqa_vlmevalkit.tsv", path_or_fileobj=visonlyqa_tsv_path)
