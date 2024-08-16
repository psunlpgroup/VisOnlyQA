import json

import numpy as np
import matplotlib.pyplot as plt

from src.config import (
    finetuning_base_models_list, base_model_to_finetuned_model_dict, finetuning_splits_dict, convert_model_name
)
from src.path import analysis_dir, figures_dir, get_evaluation_metrics_path


colors_dict = {
    "microsoft/Phi-3.5-vision-instruct": "green",
    "OpenGVLab/InternVL2-4B": "plum",
    "OpenGVLab/InternVL2-8B": "purple",
    "Qwen/Qwen2-VL-2B-Instruct": "bisque",
    "Qwen/Qwen2-VL-7B-Instruct": "darkorange",
}


def plot_model_comparison_grouped_categories_bottom(diff_dict: dict, finetuned_accuracy_dict: dict,
                                                    split_list: list[str], real_synthetic: str, add_legend: bool=True) -> None:
    fontsize = 13
    
    # Categories divided into broader groups
    major_categories = [split.split("_")[0].capitalize() for split in split_list]
    sub_categories = [split.split("__")[1].capitalize() for split in split_list]
    
    # Number of models
    num_models = len(diff_dict)
    
    # X-axis positions
    x = np.arange(len(sub_categories))
    
    # Width of the bars (dynamic based on the number of models)
    width = 0.8 / num_models
    
    for category in ["improvement", "original"]:
        # Create the plot
        fig, ax = plt.subplots(figsize=(24, 3))
        
        # Loop through each model to plot its data
        for i, (model_name, diff_list) in enumerate(diff_dict.items()):
            if category == "improvement":
                percent = [s * 100 for s in diff_list]
            else:
                percent = [s * 100 for s in finetuned_accuracy_dict[model_name]]
            
            ax.bar(x + (i - num_models / 2) * width, percent, width, label=convert_model_name[model_name], color=colors_dict[model_name])
            
            # add numbers on top of the bars
            for j, p in enumerate([s * 100 for s in diff_list]):
                x_position = p if category == "improvement" else finetuned_accuracy_dict[model_name][j] * 100
                
                if sub_categories[j] != "":
                    color = "black" if p >= 0 else "red"
                    ax.text(x[j] + (i - num_models / 2) * width, max(x_position, 0),
                            f"{p:+.1f}" if p!=0 else "0.0", ha="center", va="bottom", fontsize=fontsize-2, color=color)
                
                    finetuned_accuracies = [s * 100 for s in finetuned_accuracy_dict[model_name]]
                    ax.text(x[j] + (i - num_models / 2) * width, max(x_position, 0) - 8,
                            f"({finetuned_accuracies[j]:.0f}%)", ha="center", va="bottom", fontsize=fontsize-2, color="black")

        # Add main x-labels
        y_label = {"improvement": "Improvement by\nFine-tuning (Accuracy)", "original": "Fine-tuned Accuracy"}[category]
        ax.set_ylabel(y_label, fontsize=fontsize)
        
        scores_list = {"improvement": list(diff_dict.values()), "original": list(finetuned_accuracy_dict.values())}[category]
        if category == "improvement":
            y_min = min([min(scores) for scores in scores_list]) * 100
            y_max = max([max(scores) for scores in scores_list]) * 100 + 8
        else:
            y_min = 0
            y_max = max([max(scores) for scores in scores_list]) * 100 + 12
        ax.set_ylim(y_min, y_max)

        # Create two rows of labels
        ax.set_xticks(x)
        ax.set_xticklabels(sub_categories, fontsize=fontsize)  # , rotation=45, ha="right")

        # Create a secondary x-axis at the bottom for the major categories
        unique_categories = np.unique(major_categories)
        major_category_positions = [np.mean([i for i, cat in enumerate(major_categories) if cat == major]) 
                                    for major in unique_categories]
        
        ax_secondary = ax.secondary_xaxis('bottom')
        ax_secondary.set_xticks(major_category_positions)
        ax_secondary.set_xticklabels(unique_categories, rotation=0)

        ax_secondary.spines['bottom'].set_position(('outward', 25))
        ax_secondary.spines['bottom'].set_visible(False)
        ax_secondary.tick_params(axis='x', length=0, labelsize=fontsize)
        
        # add 0 horizontal line
        if category == "improvement":
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

        # Add legend
        if add_legend:
            ax.legend(fontsize=fontsize+2)
        
        # add title
        title = {"real": "VisOnlyQA-Eval-Real (Out-of-Distribution Figures)", "synthetic": "VisOnlyQA-Eval-Synthetic (In-Distribution Figures)"}[real_synthetic]
        ax.set_title(title, fontsize=fontsize+2)
        
        # Display the plot
        plt.tight_layout()

        figures_dir.mkdir(parents=True, exist_ok=True)
        output_image_name = {"improvement": f"improvement_by_finetuning_{real_synthetic}", "original": f"finetuned_accuracy_{real_synthetic}"}[category]
        plt.savefig(figures_dir / f"{output_image_name}.png")


if __name__ == "__main__":
    metric_name = "accuracy"
    
    table_dict: dict[str, dict[str, dict[str, float]]] = {}
    for real_synthetic in ["real", "synthetic"]:
        diff_dict = {}
        original_accuracy_dict = {}
        finetuned_accuracy_dict = {}
        
        for base_model_name in finetuning_base_models_list:
            diff_list: list[float] = []
            original_accuracy_list: list[float] = []
            finetuned_accuracy_list: list[float] = []            
            for split_idx, synthetic_split in enumerate(finetuning_splits_dict["finetuning_splits"]):
                finetuned_model = base_model_to_finetuned_model_dict[base_model_name][synthetic_split]
                split = synthetic_split if real_synthetic == "synthetic" else finetuning_splits_dict["corresponding_real_splits"][split_idx]
                
                if split is None:  # 3D size and angle do not have corresponding real splits
                    diff_list.append(0)
                    original_accuracy_list.append(0)
                    finetuned_accuracy_list.append(0)
                    continue
                
                performance_dict = {}
                for key, model_name in [["base", base_model_name], ["finetuned", finetuned_model]]:
                    metric_path = get_evaluation_metrics_path(
                        split=split, prompt="no_reasoning", model_name=model_name,
                        train_eval=f"eval_{real_synthetic}"
                    )
                    
                    if not metric_path.exists():
                        raise FileNotFoundError(f"Metrics file not found: {metric_path}")

                    with open(metric_path, "r") as f:
                        performance_dict[key] = json.load(f)
                
                diff = performance_dict["finetuned"][metric_name] - performance_dict["base"][metric_name]
                diff_list.append(diff)
                
                original_accuracy_list.append(performance_dict["base"][metric_name])
                finetuned_accuracy_list.append(performance_dict["finetuned"][metric_name])
            
            diff_dict[base_model_name] = diff_list
            finetuned_accuracy_dict[base_model_name] = finetuned_accuracy_list
            original_accuracy_dict[base_model_name] = original_accuracy_list
            
            table_dict.setdefault(base_model_name, {})[real_synthetic] = {}
            table_dict[base_model_name][real_synthetic]["original"] = np.average(original_accuracy_list) * 100
            table_dict[base_model_name][real_synthetic]["finetuned"] = np.average(finetuned_accuracy_list) * 100
        
        split_list = finetuning_splits_dict["finetuning_splits"] if real_synthetic == "synthetic" \
            else [s if s is not None else "__" for s in finetuning_splits_dict["corresponding_real_splits"]]
        
        add_legend = real_synthetic == "real"
        plot_model_comparison_grouped_categories_bottom(diff_dict, finetuned_accuracy_dict, split_list, real_synthetic=real_synthetic, add_legend=add_legend)

    # save the table
    table: list[list[str]] = [["Model", "Real", "Synthetic", "Real", "Synthetic"]]  # (Real, Synthetic) for Original and fine-tuned accuracy
    for model_name, values in table_dict.items():
        table.append([model_name] + [f"{values['real']['original']:.1f}", f"{values['synthetic']['original']:.1f}", f"{values['real']['finetuned']:.1f}", f"{values['synthetic']['finetuned']:.1f}"])
    
    table_dir = analysis_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    with open(table_dir / "finetuning_improvement_table.tex", "w") as f:
        for row in table:
            f.write(" & ".join(row) + " \\\\\n")
