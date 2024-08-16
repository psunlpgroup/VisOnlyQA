# This script generates csv files for human performance annotation interface
# We upload the csv files to Google Spreadsheet and share the link with annotators
# To show images in the interface, we upload images to Google Drive
# We use the Image() function in Google Spreadsheet to show images (the Image function is not included in the generated csv files)

import csv
import json

import datasets
from tap import Tap
from tqdm import tqdm

import pydrive
import pydrive.auth
from pydrive.drive import GoogleDrive
import gspread
from gspread_formatting import *

from src.path import test_human_performance_annotation_dir, test_image_path_dir, test_dataset_dir
from src.utils import get_hf_dataset_name
from src.config import visonlyqa_real_splits, visonlyqa_synthetic_splits, eval_real_splits_capitalized, eval_synthetic_splits_capitalized
from src.evaluation_dataset.dataset_annotation.create_annotation_spreadsheet import get_gdrive_url_for_figure_or_upload_figure


class HumanAnnotationInterfeaceTap(Tap):
    upload_images_to_gdrive: bool = False


annotators_num = 3

if __name__ == "__main__":
    args = HumanAnnotationInterfeaceTap().parse_args()
    
    # this is only for the synthetic images
    # if you run this code for the first time, synthetic images are not uploaded to google drive
    # so you need to set --upload_images_to_gdrive to True
    if args.upload_images_to_gdrive:
        gauth = pydrive.auth.GoogleAuth()
        gauth.LocalWebserverAuth()
        drive = GoogleDrive(gauth)
    
    splits_dir = {"real": visonlyqa_real_splits, "synthetic": visonlyqa_synthetic_splits}
    capitalized_dir = {"real": eval_real_splits_capitalized, "synthetic": eval_synthetic_splits_capitalized}
    
    annotation_num_each_split = 10

    test_human_performance_annotation_spreadsheet = test_human_performance_annotation_dir / "spreadsheet"
    test_human_performance_annotation_spreadsheet.mkdir(parents=True, exist_ok=True)
    
    for real_synthetic in ["real", "synthetic"]:
        print(f"Processing {real_synthetic}")
        
        repository_name = get_hf_dataset_name(f"eval_{real_synthetic}")
        
        # load image_url_dict -- dict[image_path] = image_url in gdrive
        image_url_path = test_image_path_dir / f"{real_synthetic}_image_url.json"
        if image_url_path.exists():
            with open(image_url_path, "r") as f:
                image_url_dict = json.load(f)
        else:
            image_url_dict = {}
        
        image_url_dict_updated = False
        
        # create annotation spreadsheet
        dataset_stats: dict = {}
        for split in splits_dir[real_synthetic]:
            dataset = datasets.load_dataset(repository_name, split=split)
            
            if len(dataset) < 2 + annotation_num_each_split * annotators_num:
                print(f"Skip {split} due to insufficient data")
                continue
            
            dataset_shuffled = dataset.shuffle(seed=22)
            
            data_idx = 0
            examples = []
            for _ in range(2):
                d = dataset_shuffled[data_idx]
                examples.append(d)
                data_idx += 1
            
            for annotators_id in range(annotators_num):
                print(f"Processing {split} annotators {annotators_id}")
                
                data_for_this_annotator = []
                for _ in range(annotation_num_each_split):
                    d = dataset_shuffled[data_idx]
                    data_for_this_annotator.append(d)
                    data_idx += 1
                data_for_this_annotator = examples + data_for_this_annotator
                
                annotation_rows = []
                for d in tqdm(data_for_this_annotator):
                    image_path = d["image_path"]
                    
                    if image_path in image_url_dict.keys():
                        image_url = image_url_dict[image_path]
                    elif not args.upload_images_to_gdrive:
                        raise ValueError(f"Image {image_path} does not exist in image_url_dict. If you want to upload images to Google Drive, set --upload_images_to_gdrive to True")
                    else:  # args.upload_images_to_gdrive is True
                        image_url = get_gdrive_url_for_figure_or_upload_figure(drive, image_path, local_image_dir=test_dataset_dir, drive_dir_name="VisOnlyQA_annotation")
                        image_url_dict[image_path] = image_url
                        image_url_dict_updated = True
                    
                    question = d["question"]
                    answer = d["answer"]
                    data_id = d["id"]
                    
                    annotation_rows.append([data_id, image_url, "", question, answer])
                
                with open(test_human_performance_annotation_spreadsheet / f"{split}_{annotators_id}.csv", "w") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(annotation_rows)
                
                if image_url_dict_updated:
                    with open(image_url_path, "w") as f:
                        json.dump(image_url_dict, f)
