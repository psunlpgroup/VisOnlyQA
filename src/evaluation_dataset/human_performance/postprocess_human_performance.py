from typing import Literal
import csv
import json

import numpy as np

from src.config import visonlyqa_real_splits, visonlyqa_synthetic_splits, visonlyqa_response_type_dir
from src.path import test_human_performance_annotation_dir, human_performance_dir
from src.evaluation_dataset.human_performance.generate_human_performance_annotation_interface import annotators_num



def convert_annotation_to_y_pred(annotation: list[Literal["TRUE", "FALSE"]], split: str):
    options = visonlyqa_response_type_dir[split][1]
    
    true_indices = [i for i, a in enumerate(annotation) if a == "TRUE"]
    y_pred = ",".join([options[i] for i in true_indices])
    
    return y_pred


if __name__ == "__main__":
    human_performance_dir.mkdir(parents=True, exist_ok=True)
    
    for real_synthetic in ["real", "synthetic"]:
        splits_list = {"real": visonlyqa_real_splits, "synthetic": visonlyqa_synthetic_splits}[real_synthetic]

        human_performance = {}
        for annotator_id in range(annotators_num):
            for split in splits_list:
                annotation_file = test_human_performance_annotation_dir / "annotated" / f"{split}_{annotator_id}.csv"
                
                if not annotation_file.exists():
                    continue
                
                y_true: list[str] = []
                y_pred: list[str] = []
                with open(annotation_file, "r") as f:
                    reader = csv.reader(f)
                    
                    # skip the first 3 rows (header)
                    for _ in range(3):
                        next(reader)
                    
                    for row in reader:
                        selected_options = row[5:]
                        if not any([o == "TRUE" for o in selected_options]):  # not annotated yet
                            continue
                        
                        y_true.append(row[4])
                        y_pred.append(convert_annotation_to_y_pred(selected_options, split=split))
                
                correct_list = [1 if yt.lower() == yp.lower() else 0 for yt, yp in zip(y_true, y_pred)]
                human_performance[str((split, annotator_id))] = {
                    "accuracy": np.mean(correct_list).item(),
                    "annotations": {"y_true": y_true, "y_pred": y_pred, "correct_list": correct_list},
                }
        
        # average over all annotators
        for split in splits_list:
            human_performance[str((split, "average"))] = {
                "accuracy": np.mean([human_performance[str((split, annotator_id))]["accuracy"] for annotator_id in range(annotators_num) if str((split, annotator_id)) in human_performance]).item(),
            }
        
        human_performance[str(("all", "average"))] = {
            "accuracy": np.mean([human_performance[str((split, "average"))]["accuracy"] for split in splits_list]).item(),
        }
        
        with open(human_performance_dir / f"{real_synthetic}_human_performance.json", "w") as f:
            json.dump(human_performance, f, indent=4)
