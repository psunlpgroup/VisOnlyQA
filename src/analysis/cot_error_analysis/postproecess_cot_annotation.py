import csv
import json

import matplotlib.pyplot as plt

from src.config import convert_model_name
from src.analysis.cot_error_analysis.generate_cot_annotation_spreadsheet import cot_error_intermediate_dir, cot_error_analysis_dir, cot_error_analysis_models_list


cot_error_labels_list = ["Correct", "Question Understanding Error", "Perception Error", "Vague Reasoning", "Redundant Reasoning", "Reasoning Error"]

colors_dict = {
    "Correct": "tab:blue",
    "Question Understanding Error": "tab:orange",
    "Perception Error": "tab:purple",
    "Vague Reasoning": "tab:pink",
    "Redundant Reasoning": "tab:pink",
    "Reasoning Error": "tab:red"
}


def convert_error_label(error_label: str) -> str:
    if error_label == "Perception Error":
        return "Visual Perception Error"
    elif error_label == "Vague Reasoning":
        return "Insufficient Reasoning"
    else:
        return error_label


if __name__ == "__main__":
    annotation_results_all = {}
    num_annotations_for_each_split = 5

    annotation_results_save_dir = cot_error_analysis_dir / "annotation_results"
    annotation_results_save_dir.mkdir(parents=True, exist_ok=True)
    
    fontsize = 12
    
    for model_name in cot_error_analysis_models_list:
        short_model_name = model_name.split("/")[-1]
        
        for correct_incorrect in ["correct", "incorrect"]:
            error_counts: dict[str, dict[str, int]] = {}
            split_counts: dict[str, int] = {}
            
            annotaiton_file_path = cot_error_intermediate_dir / "annotations" / f"cot_error_analysis - {correct_incorrect}_{short_model_name}.csv"
            if annotaiton_file_path.exists():
                with open(annotaiton_file_path, "r") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    
                    for row in reader:
                        # if no annotation, skip
                        row_dict = dict(zip(header, row))
                        if not any([row_dict[error_label] == "TRUE" for error_label in cot_error_labels_list]):
                            continue
                        
                        # we only use first num_annotations_for_each_split annotations for each split
                        split = row[0]
                        if split in split_counts and split_counts[split] == num_annotations_for_each_split:
                            continue
                        
                        # count
                        split_counts.setdefault(split, 0)
                        split_counts[split] += 1
                        split_counts.setdefault("all", 0)
                        split_counts["all"] += 1
                        
                        # store annotations
                        for error_label in cot_error_labels_list:
                            if row_dict[error_label] == "TRUE":
                                error_counts.setdefault(split, {}).setdefault(error_label, 0)
                                error_counts[split][error_label] += 1
                                error_counts.setdefault("all", {}).setdefault(error_label, 0)
                                error_counts["all"][error_label] += 1
            else:
                print(f"{annotaiton_file_path} does not exist.")
            
            annotation_results_all.setdefault(short_model_name, {})[correct_incorrect] = {"error_counts": error_counts, "split_counts": split_counts}
            
            with open(annotation_results_save_dir / f"{short_model_name}_{correct_incorrect}_annotations.json", "w") as f:
                json.dump(annotation_results_all[short_model_name][correct_incorrect], f, indent=4)
            
    ###
    # bar plot
    plt.rcParams.update({'font.size': fontsize})
    
    # figure with four subplots
    for figure_type in ["all", "incorrect"]:
        cot_error_labels_to_show = cot_error_labels_list if figure_type == "all" else cot_error_labels_list[1:]
        
        num_columns = 2 if figure_type == "all" else 1
        fig, axs = plt.subplots(
            len(cot_error_analysis_models_list), num_columns,
            figsize=(8 * num_columns, 1.6 * len(cot_error_analysis_models_list))
        )
        for model_idx, model_name in enumerate(cot_error_analysis_models_list):
            short_model_name = model_name.split("/")[-1]

            final_answer_types_list = ["correct", "incorrect"] if figure_type == "all" else ["incorrect"]
            for ci_idx, correct_incorrect in enumerate(final_answer_types_list):
                try:
                    total_num = annotation_results_all[short_model_name][correct_incorrect]["split_counts"]["all"]
                    error_counts = annotation_results_all[short_model_name][correct_incorrect]["error_counts"]
                except KeyError:
                    total_num = 1
                    error_counts = {"all": {error_label: 0 for error_label in cot_error_labels_to_show}}
                
                error_percents_list = [
                    error_counts["all"][error_label] / total_num * 100 if error_label in error_counts["all"] else 0
                    for error_label in cot_error_labels_to_show
                ]
                
                ax: plt.Axes = axs[model_idx, ci_idx] if figure_type == "all" else axs[model_idx]
                
                cot_error_labels_list_with_line_break = [convert_error_label(label).replace(" ", "\n") for label in cot_error_labels_to_show]
                colors = [colors_dict[e] for e in cot_error_labels_to_show]
                ax.bar(cot_error_labels_list_with_line_break, error_percents_list, color=colors)
                
                # number labels of error percents
                for i, error_percent in enumerate(error_percents_list):
                    ax.text(i, error_percent, f"{error_percent:.0f}%", ha="center", va="bottom", fontsize=fontsize-1)
                
                if correct_incorrect == final_answer_types_list[0]:
                    # add model name to the left side of y label (outside of the plot) with larger font size
                    model_name_converted = convert_model_name[model_name]
                    # don't make % larger
                    ax.set_ylabel(f"{model_name_converted}", fontsize=fontsize+2)
                    
                else:
                    ax.set_yticklabels([])
                
                if model_name == cot_error_analysis_models_list[0] and figure_type == "all":
                    ax.set_title(f"Errors in CoT when Final Answer is {correct_incorrect.capitalize()} [%]", fontsize=fontsize+2)

                # remove x label
                if short_model_name != cot_error_analysis_models_list[-1]:
                    ax.set_xticklabels([])
                            
                ax.set_ylim(0, 115)

            plt.tight_layout()
            
            if figure_type == "all":
                fig_name = "cot_error_annotation_results.png"
            else:
                fig_name = f"cot_error_annotation_results_incorrect.png"
            figure_path = cot_error_analysis_dir / fig_name
            plt.savefig(figure_path)
