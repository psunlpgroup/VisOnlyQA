import json

from src.typing import QTYPE
from src.config import (
    models_list, open_models_list,
    visonlyqa_real_splits, eval_real_splits_capitalized, visonlyqa_synthetic_splits, eval_synthetic_splits_capitalized,
    model_row_colors,
    convert_model_name, visonlyqa_response_type_dir,
)
from src.path import (
    get_evaluation_metrics_path, tables_dir,
    eval_real_dataset_stats_path, eval_synthetic_dataset_stats_path,
    human_performance_dir,
)


def get_random_baseline_performance(question_type: QTYPE, options_num: int, metric: str="accuracy") -> float:
    if metric == "accuracy":
        if question_type == "single_answer":
            return 1 / options_num
        else:
            assert question_type == "multiple_answers"
            return 1 / (2 ** options_num)
    else:
        raise NotImplementedError(f"metric={metric} is not implemented")


def row_to_str(row: list[str]):
    if len(row) == 1 and "rule" in row[0]:
        return row[0] + "\n"
    else:
        return " & ".join(row) + " \\\\\n"


if __name__ == "__main__":
    for real_synthetic in ["real", "synthetic"]:
        table_sub_dir = tables_dir / real_synthetic
        table_sub_dir.mkdir(parents=True, exist_ok=True)
        
        # get dataset stats
        eval_dataset_stats_path = {"real": eval_real_dataset_stats_path, "synthetic": eval_synthetic_dataset_stats_path}[real_synthetic]
        with open(eval_dataset_stats_path, "r") as f:
            dataset_stats = json.load(f)
        
        # human performance
        human_performance_path = human_performance_dir / f"{real_synthetic}_human_performance.json"
        if human_performance_path.exists():
            with open(human_performance_path, "r") as f:
                human_performance: dict[str, dict[str, float]] = json.load(f)
        else:
            human_performance = None
        
        # splits
        splits_list = {"real": visonlyqa_real_splits, "synthetic": visonlyqa_synthetic_splits}[real_synthetic]
        splits_capitalized = {"real": eval_real_splits_capitalized, "synthetic": eval_synthetic_splits_capitalized}[real_synthetic]
        
        # performance
        for prompt_type in ["no_reasoning", "reasoning"]:
            for metric_name in ["accuracy"]:
                table = []
                first_row = [""] + splits_capitalized + [""]  # Average
                table.append(first_row)
                table.append(["\\midrule"])
                
                # random baseline
                row = ["Random"]
                random_performance_list = []
                for split in splits_list:
                    question_type, options = visonlyqa_response_type_dir[split]
                    value = get_random_baseline_performance(question_type=question_type, options_num=len(options), metric=metric_name)
                    row.append(f"{value*100:.1f}")
                    random_performance_list.append(value)
                
                split_ratio_list = [dataset_stats[split]["num_examples"] / dataset_stats["all"]["num_examples"] for split in splits_list]
                average = sum([random_performance_list[i] * split_ratio_list[i] for i in range(len(splits_list))]) * 100
                row.append(f"{average:.1f}")
                
                table.append(row)
                table.append(["\\midrule"])
                
                # model performance
                for model_name in models_list:
                    model_name_str = convert_model_name[model_name]
                    if model_name in model_row_colors.keys():
                        model_name_str = f"\\rowcolor{{{model_row_colors[model_name]}!20}}{model_name_str}"
                    
                    row = [model_name_str]
                    for split in splits_list + ["all"]:
                        metrics_path = get_evaluation_metrics_path(split=split, prompt=prompt_type, model_name=model_name, train_eval=f"eval_{real_synthetic}")
                        
                        if metrics_path.exists():
                            with open(metrics_path, "r") as f:
                                metrics = json.load(f)
                            
                            metric_string = f"{metrics[metric_name]*100:.1f}"
                        else:
                            metric_string = ""
                        row.append(metric_string)
                    table.append(row)
                    
                    if model_name == open_models_list[-1] or model_name == models_list[-1]:
                        table.append(["\\midrule"])
                table.append(["\\midrule"])
                
                # human performance
                row = ["Human"]
                if human_performance is not None:
                    for split in splits_list + ["all"]:
                        key = str((split, "average"))
                        if key in human_performance.keys():
                            value = human_performance[key][metric_name] * 100
                            metric_string = f"{value:.1f}"
                        else:
                            metric_string = ""
                        row.append(metric_string)
                table.append(row)            

                # write latex table
                table_path = table_sub_dir / f"prompt={prompt_type},metric={metric_name}.txt"
                with open(table_path, "w") as f:
                    for row in table:
                        f.write(row_to_str(row))
        
        # diff table
        for prompt_1, prompt_2 in [["no_reasoning", "reasoning"]]:
            for metric_name in ["accuracy"]:
                table = []
                first_row = [""] + splits_capitalized + [""]  # Average
                table.append(first_row)
                table.append(["\\midrule"])
                
                for model_name in models_list:
                    row = [convert_model_name[model_name]]
                    for split in splits_list + ["all"]:
                        metrics_path_1 = get_evaluation_metrics_path(split=split, prompt=prompt_1, model_name=model_name, train_eval=f"eval_{real_synthetic}")
                        metrics_path_2 = get_evaluation_metrics_path(split=split, prompt=prompt_2, model_name=model_name, train_eval=f"eval_{real_synthetic}")
                        
                        if metrics_path_1.exists() and metrics_path_2.exists():
                            with open(metrics_path_1, "r") as f:
                                metrics_1 = json.load(f)
                            with open(metrics_path_2, "r") as f:
                                metrics_2 = json.load(f)
                            
                            diff = (metrics_2[metric_name] - metrics_1[metric_name]) * 100
                            color = "red" if diff < 0 else "teal" if diff > 0 else "gray"  # diff == 0
                            
                            row.append(f"\\textcolor{{{color}}}{{{diff:.1f}}}")
                        else:
                            row.append("")
                    table.append(row)
                    if model_name == open_models_list[-1]:
                        table.append(["\\midrule"])

                # write latex table
                table_path = table_sub_dir / f"prompt_diff={prompt_1}-{prompt_2},metric={metric_name}.txt"
                with open(table_path, "w") as f:
                    for row in table:
                        f.write(row_to_str(row))
