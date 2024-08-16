import csv
import json
import random

import datasets

from src.path import test_image_path_dir, analysis_dir, test_intermediate_dir, get_evaluation_model_responses_path
from src.utils import get_hf_dataset_name
from src.config import visonlyqa_real_splits, visonlyqa_synthetic_splits, eval_real_splits_capitalized, eval_synthetic_splits_capitalized


cot_error_analysis_dir = analysis_dir / "cot_error_analysis"

cot_error_intermediate_dir = test_intermediate_dir / "cot_error_analysis"
cot_error_annotations_dir = cot_error_intermediate_dir / "annotation_csv"

cot_error_analysis_models_list = [
    "OpenGVLab/InternVL2-4B", "OpenGVLab/InternVL2-8B", "OpenGVLab/InternVL2-26B", "OpenGVLab/InternVL2-Llama3-76B", 
    # "microsoft/Phi-3.5-vision-instruct",
    "gpt-4o-2024-08-06",
    "gemini-1.5-pro-002",
]


if __name__ == "__main__":
    splits_dir = {"real": visonlyqa_real_splits, "synthetic": visonlyqa_synthetic_splits}
    capitalized_dir = {"real": eval_real_splits_capitalized, "synthetic": eval_synthetic_splits_capitalized}
    
    annotation_num_each_split = 5

    cot_error_analysis_dir.mkdir(parents=True, exist_ok=True)
    cot_error_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    with open(test_image_path_dir / "real_image_url.json", "r") as f:
        image_url_dict = json.load(f)
    
    cot_stats = {}
    
    for real_synthetic in ["real", "synthetic"]:
        if real_synthetic == "synthetic":
            continue  # not implemented yet
        
        repository_name = get_hf_dataset_name(f"eval_{real_synthetic}")
        for model_name in cot_error_analysis_models_list:
            print(f"Processing {real_synthetic} split for {model_name}")
            
            model_all_rows_dict: dict[str, list] = {}

            cot_stats.setdefault(model_name, {})[real_synthetic] = {}
            
            for split in splits_dir[real_synthetic]:
                print(f"Processing {split}")
                
                dataset = datasets.load_dataset(repository_name, split=split)
                selected_indices = random.Random(68).sample(list(range(len(dataset))), len(dataset))[:annotation_num_each_split]

                # load model responses
                model_responses_dict = {}
                for prompt in ["reasoning", "no_reasoning"]:
                    model_responses_path = get_evaluation_model_responses_path(
                        split=split, prompt=prompt, model_name=model_name, train_eval=f"eval_{real_synthetic}"
                    )
                    with open(model_responses_path, "r") as f:
                        model_responses_dict[prompt] = [json.loads(line) for line in f]
                
                # this does not happen in the final dataset
                if len(dataset) < annotation_num_each_split:
                    print(f"Skip {split} due to insufficient data")
                    continue
                
                annotation_rows_all = []
                for data_idx in range(len(dataset)):
                    d = dataset[data_idx]
                    r = model_responses_dict["reasoning"][data_idx]
                    
                    image_path = d["image_path"]
                    image_url = image_url_dict[image_path]  # uploaded to google drive when creating annotation spreadsheet
                    
                    question = d["question"]
                    answer = d["answer"]
                    data_id = d["id"]
                    
                    assert question in r["prompt"]  # make sure that responses correctly corresponds to the question
                    
                    cot = r["response"]
                    reasoning_y_pred = r["y_pred"]
                    no_reasoning_y_pred = model_responses_dict["no_reasoning"][data_idx]["y_pred"]
                    
                    # improved or degraded by chain-of-thought
                    reasoning_y_pred_correct = "correct" if answer == reasoning_y_pred else "incorrect"
                    if reasoning_y_pred == no_reasoning_y_pred:
                        cot_changed_y_pred = "same"
                    elif reasoning_y_pred_correct == "correct":  # reasoning_y_pred != no_reasoning_y_pred
                        cot_changed_y_pred = "improved"
                    else:  # reasoning_y_pred != no_reasoning_y_pred
                        cot_changed_y_pred = "degraded"

                    annotation_rows_all.append([split, data_id, image_url, "", question, answer, reasoning_y_pred, reasoning_y_pred_correct, cot_changed_y_pred, cot])
                
                # stats
                cot_stats[model_name][real_synthetic][split] = {}

                # randomly selected
                random_annotation_rows = [annotation_rows_all[i] for i in selected_indices]
                model_all_rows_dict.setdefault("random", []).extend(random_annotation_rows)
                
                # improved or degraded
                for im_dg in ["improved", "degraded"]:
                    improved_degraded_annotation_rows = [r for r in annotation_rows_all if r[-2] == im_dg]
                    model_all_rows_dict.setdefault(im_dg, []).extend(improved_degraded_annotation_rows)
                    
                    # stats
                    cot_stats[model_name][real_synthetic][split][f"%_{im_dg}"] = len(improved_degraded_annotation_rows) / len(annotation_rows_all) * 100
                
                # correct or incorrect
                for c_i in ["correct", "incorrect"]:
                    correct_incorrect_annotation_rows = [r for r in annotation_rows_all if r[-3] == c_i]
                    
                    selected_rows = random.Random(68).sample(correct_incorrect_annotation_rows, len(correct_incorrect_annotation_rows))[:annotation_num_each_split]
                    model_all_rows_dict.setdefault(c_i, []).extend(selected_rows)
                    
                    # stats
                    cot_stats[model_name][real_synthetic][split][f"%_{c_i}"] = len(correct_incorrect_annotation_rows) / len(annotation_rows_all) * 100
            
            for key, value in model_all_rows_dict.items():
                model_name_short = model_name.split("/")[-1]
                
                with open(cot_error_annotations_dir / f"{key}_{model_name_short}.csv", "w") as f:
                    csv_writer = csv.writer(f)
                    csv_writer.writerows(value)
    
    with open(cot_error_analysis_dir / "cot_stats.json", "w") as f:
        json.dump(cot_stats, f, indent=4)
