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
    repository_name = get_hf_dataset_name("metadata")
    
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
    
    # upload meta data
    api.upload_file(
        repo_type="dataset",
        repo_id=repository_name,
        path_or_fileobj="./intermediate/alphageometry/info.tar.gz",
        path_in_repo="metadata.tar.gz",
    )

    # upload meta data
    api.upload_file(
        repo_type="dataset",
        repo_id=repository_name,
        path_or_fileobj="./intermediate/alphageometry/images.tar.gz",
        path_in_repo="images.tar.gz",
    )
