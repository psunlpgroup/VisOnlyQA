import json

from src.config import (
    visonlyqa_real_splits, eval_real_splits_capitalized, visonlyqa_synthetic_splits, eval_synthetic_splits_capitalized,
    visonlyqa_synthetic_with_text_splits, eval_synthetic_with_text_splits_capitalized,
    finetuning_splits_dict,
    convert_model_name, visonlyqa_response_type_dir,
    finetuning_base_models_list, finetuning_splits_capitalized, base_model_to_finetuned_model_dict
)
from src.path import get_evaluation_metrics_path, tables_dir, eval_real_dataset_stats_path, eval_synthetic_dataset_stats_path, eval_synthetic_with_text_dataset_stats_path
from src.evaluation.generate_tables import get_random_baseline_performance, row_to_str


if __name__ == "__main__":
    table_sub_dir = tables_dir / "finetuned"
    table_sub_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_stats = {}
    for real_synthetic in ["real", "synthetic", "synthetic_with_text"]:
        # get dataset stats
        eval_dataset_stats_path = {"real": eval_real_dataset_stats_path, "synthetic": eval_synthetic_dataset_stats_path, "synthetic_with_text": eval_synthetic_with_text_dataset_stats_path}[real_synthetic]
        with open(eval_dataset_stats_path, "r") as f:
            dataset_stats[real_synthetic] = json.load(f)
    
    splits_list_dir = {"real": visonlyqa_real_splits, "synthetic": visonlyqa_synthetic_splits, "synthetic_with_text": visonlyqa_synthetic_with_text_splits}
    splits_capitalized_dir = {"real": eval_real_splits_capitalized, "synthetic": eval_synthetic_splits_capitalized, "synthetic_with_text": eval_synthetic_with_text_splits_capitalized}
    
    # finetuning table
    prompt_type = "no_reasoning"
    for metric_name in ["accuracy"]:
        for splits_list in [["finetuning_splits", "corresponding_real_splits"], ["finetuning_text_splits"]]:
            table = []
            first_row = ["", ""] + finetuning_splits_capitalized
            table.append(first_row)
            table.append(["\\midrule"])

            for splits_name in splits_list:
                real_synthetic = {
                    "finetuning_splits": "synthetic",
                    "corresponding_real_splits": "real",
                    "finetuning_text_splits": "synthetic_with_text"
                }[splits_name]
                
                # first column
                rows_num = 1 + 2 * len(finetuning_base_models_list)
                row = {
                    "real": ["\\multirow{" + str(rows_num) + "}{*}{Eval-Real (Out-of-Domain)}"],
                    "synthetic": ["\\multirow{" + str(rows_num) + "}{*}{Eval-Synthetic (In-Domain)}"],
                    "synthetic_with_text": ["\\multirow{" + str(rows_num) + "}{*}{Eval-Synthetic with Text (In-Domain)}"]
                }[real_synthetic]
                
                # random baseline
                row.append("\\multicolumn{2}{c}{Random}")
                random_performance_list = []
                for split in finetuning_splits_dict[splits_name]:
                    if split is not None:
                        question_type, options = visonlyqa_response_type_dir[split]
                        value = get_random_baseline_performance(question_type=question_type, options_num=len(options), metric=metric_name)
                        row.append(f"{value*100:.1f}")
                    else:  # split is None (3d data does not hvae corresponding real data)
                        row.append("--")
                table.append(row)
                
                table.append(["\\cmidrule{2-11}"])
                
                # model performance
                for model_name in finetuning_base_models_list:
                    # first row: original model
                    row = ["", "\\multirow{2}{*}{" + convert_model_name[model_name] + "}", "Original"]
                    metrics_list = []
                    for split in finetuning_splits_dict[splits_name]:
                        if split is not None:
                            metrics_path = get_evaluation_metrics_path(split=split, prompt=prompt_type, model_name=model_name, train_eval=f"eval_{real_synthetic}")
                            
                            if metrics_path.exists():
                                with open(metrics_path, "r") as f:
                                    metrics = json.load(f)
                                
                                metric = metrics[metric_name]
                                metric_string = f"{metric*100:.1f}"
                                metrics_list.append(metric)
                            else:
                                metric_string = ""
                        else:  # split is None (3d data does not hvae corresponding real data)
                            metric_string = "--"
                        row.append(metric_string)
                    
                    if len(metrics_list) > 0:
                        average_string = f"{sum(metrics_list) / len(metrics_list) * 100:.1f}"
                    else:
                        average_string = ""
                    row.append(average_string)
                    
                    table.append(row)
                    
                    # second row: finetuned model
                    row = ["", "", "Fine-tuned"]
                    metrics_list = []
                    for split in finetuning_splits_dict[splits_name]:
                        if split is not None:
                            if splits_name == "corresponding_real_splits":
                                split_key = "synthetic" + split  # syntheticgeometry__{name}
                            else:
                                split_key = split
                            
                            if split_key not in base_model_to_finetuned_model_dict[model_name].keys():
                                row.append("")
                                continue
                            
                            finetuned_model_name = base_model_to_finetuned_model_dict[model_name][split_key]
                            metrics_path = get_evaluation_metrics_path(split=split, prompt=prompt_type, model_name=finetuned_model_name, train_eval=f"eval_{real_synthetic}")

                            if metrics_path.exists():
                                with open(metrics_path, "r") as f:
                                    metrics = json.load(f)
                                
                                metric = metrics[metric_name]
                                metric_string = f"{metric*100:.1f}"
                                metrics_list.append(metric)
                            else:
                                metric_string = ""
                        else:  # split is None (3d data does not hvae corresponding real data)
                            metric_string = "--"
                        row.append(metric_string)
                    
                    if len(metrics_list) > 0:
                        average_string = f"{sum(metrics_list) / len(metrics_list) * 100:.1f}"
                    else:
                        average_string = ""
                    row.append(average_string)
                        
                    table.append(row)
                    
                    table.append(["\\cmidrule{2-11}"])
            
            # write latex table
            if splits_list[-1] == "corresponding_real_splits":
                table_path = table_sub_dir / f"fine-tuned--prompt={prompt_type},metric={metric_name}.txt"
            else:
                table_path = table_sub_dir / f"fine-tuned-with-text--prompt={prompt_type},metric={metric_name}.txt"
            with open(table_path, "w") as f:
                for row in table:
                    f.write(row_to_str(row))
