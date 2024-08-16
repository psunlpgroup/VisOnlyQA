from pathlib import Path
import shutil

import datasets
from functools import partial
from huggingface_hub import HfApi

from src.path import train_dataset_dir
from src.config import train_data_splits
from src.utils import get_hf_dataset_name
from src.upload_to_huggingface.upload_to_huggingface_test_set import add_image_to_example, read_hf_readme


if __name__ == "__main__":
    repository_name = get_hf_dataset_name("train")
    
    ###
    # upload files
    api = HfApi()
    
    # create repository
    api.create_repo(repo_id=repository_name, private=True, repo_type="dataset", exist_ok=True)
    
    # upload license
    api.upload_file(repo_type="dataset", repo_id=repository_name, path_in_repo="LICENSE.md", path_or_fileobj=Path("LICENSE.md"))
    
    # upload README
    readme = read_hf_readme(size_category="1K<n<10K")
    api.upload_file(repo_type="dataset", repo_id=repository_name, path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8"))

    # upload README images
    api.upload_folder(
        repo_type="dataset",
        repo_id=repository_name,
        folder_path="readme_figures",
        path_in_repo="readme_figures",
    )
    
    # upload text data
    for split in train_data_splits:
        split_name = f"{split}"
        dataset = datasets.load_dataset("./dataset/train/synthetic", data_files={split_name: f"{split_name}.jsonl"}, split=split_name)
        
        # add images to dataset
        dataset = dataset.map(partial(add_image_to_example, split=split_name))
        dataset.push_to_hub(repository_name, split=split_name, private=True)

    # upload converted instances
    for folder_path, path_in_repo in [
            ("dataset/train/phi35v_train_data", "phi35v_train_data"),
            ("dataset/train/sharegpt_train_data", "sharegpt_train_data"),  # qwen2-vl
            ("dataset/train/internvl_meta", "internvl_meta"), ("dataset/train/internvl_annotations", "internvl_annotations"),
            ]:
        api.upload_folder(
            repo_type="dataset",
            repo_id=repository_name,
            folder_path=folder_path,
            path_in_repo=path_in_repo,
        )

    # upload images
    train_image_dir = train_dataset_dir / "images"
    for dataset_name in ["CLEVR", "SuperCLEVR", "SyntheticGeometry"]:
        zip_file_path = train_image_dir / f"{dataset_name}.zip"
        if not zip_file_path.exists():
            print(f"Creating zip file for {dataset_name}")
            shutil.make_archive(str(zip_file_path.with_suffix("")), "zip", zip_file_path.parent, dataset_name)
        
        print(f"Uploading images for {dataset_name}")
        api.upload_file(repo_type="dataset", repo_id=repository_name, path_in_repo=f"images/{dataset_name}.zip", path_or_fileobj=zip_file_path)
