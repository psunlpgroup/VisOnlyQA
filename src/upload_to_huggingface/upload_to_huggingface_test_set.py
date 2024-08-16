from pathlib import Path

import datasets
import PIL.Image
from functools import partial
from huggingface_hub import HfApi

from src.path import test_dataset_dir, train_dataset_dir
from src.config import visonlyqa_real_splits, visonlyqa_synthetic_splits, visonlyqa_synthetic_with_text_splits
from src.utils import get_hf_dataset_name


def read_hf_readme(size_category="n<1K"):
    with open("src/upload_to_huggingface/hf_readme_metadata.md", "r") as f:
        readme = f.read().format(size_category=size_category)
    with open("src/upload_to_huggingface/hf_readme.md", "r") as f:
        readme += f.read()
    
    return readme


def add_image_to_example(example, split="test"):
    dataset_dir = test_dataset_dir if split == "test" else train_dataset_dir
    
    image_path = example["image_path"]
    image = PIL.Image.open(dataset_dir / image_path)
    new_example = {"decoded_image": image}
    new_example.update(example)
    return new_example


if __name__ == "__main__":
    for real_synthetic in ["synthetic", "synthetic_with_text", "real"]:
        repository_name = get_hf_dataset_name(f"eval_{real_synthetic}")
        
        ###
        # upload files
        api = HfApi()
        
        # create repository
        api.create_repo(repo_id=repository_name, private=True, repo_type="dataset", exist_ok=True)
        
        # upload license
        api.upload_file(repo_type="dataset", repo_id=repository_name, path_in_repo="LICENSE.md", path_or_fileobj=Path("LICENSE.md"))
        
        # upload README
        readme = read_hf_readme(size_category="n<1K")
        api.upload_file(repo_type="dataset", repo_id=repository_name, path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8"))

        # upload README images
        api.upload_folder(
            repo_type="dataset",
            repo_id=repository_name,
            folder_path="readme_figures",
            path_in_repo="readme_figures",
        )
        
        ###
        # upload dataset
        splits_list = {"synthetic": visonlyqa_synthetic_splits, "synthetic_with_text": visonlyqa_synthetic_with_text_splits, "real": visonlyqa_real_splits}[real_synthetic]
        for split_name in splits_list:
            print(split_name)
            
            dataset = datasets.load_dataset(f"./dataset/test/{real_synthetic}", data_files={split_name: f"{split_name}.jsonl"}, split=split_name)
            
            # add images to dataset
            dataset = dataset.map(partial(add_image_to_example, split="test"))
            
            dataset.push_to_hub(repository_name, split=split_name, private=True)
