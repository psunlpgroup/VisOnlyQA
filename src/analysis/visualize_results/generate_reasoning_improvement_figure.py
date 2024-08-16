import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src.config import visonlyqa_real_splits, eval_real_splits_capitalized, convert_model_name
from src.path import figures_dir, get_evaluation_metrics_path


def plot_model_comparison_grouped_categories_bottom(diff_dict: dict) -> None:
    # Categories divided into broader groups
    major_categories = [split.split("_")[0].capitalize() for split in visonlyqa_real_splits]
    sub_categories = eval_real_splits_capitalized
    
    # Number of models
    num_models = len(diff_dict)
    
    # X-axis positions
    x = np.arange(len(sub_categories))
    
    # Width of the bars (dynamic based on the number of models)
    width = 0.8 / num_models
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 3))
    
    # Generate distinct colors for each model
    colors = matplotlib.colormaps.get_cmap('tab10')
    
    # Loop through each model to plot its data
    for i, (model_name, scores) in enumerate(diff_dict.items()):
        ax.bar(x + (i - num_models / 2) * width, scores, width, label=convert_model_name[model_name], color=colors(i))
        
        # Plot average line for each model
        model_avg = np.mean(scores)
        ax.axhline(y=model_avg, color=colors(i), linestyle='dotted', label=f'Avg: {model_avg:.1f}')  #  {convert_model_name[model_name]:<20} 
    
    fontsize = 13
    
    # Add main x-labels
    ax.set_ylabel('Improvement by\nChain-of-Thought (Acc)', fontsize=fontsize)

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
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)

    # Add legend
    ax.legend(ncol=8, fontsize=fontsize)
    
    # Display the plot
    plt.tight_layout()
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "improvement_by_cot.png")


if __name__ == "__main__":
    diff_dict = {}
    
    metric_name = "accuracy"
    for model_name in ["OpenGVLab/InternVL2-Llama3-76B", "claude-3-5-sonnet-20240620", "gpt-4o-2024-08-06", "gemini-1.5-pro-002"]:
        diff_list: list[float] = []
        for split in visonlyqa_real_splits:
            metrics_path_1 = get_evaluation_metrics_path(split=split, prompt="no_reasoning", model_name=model_name, train_eval="eval_real")
            metrics_path_2 = get_evaluation_metrics_path(split=split, prompt="reasoning", model_name=model_name, train_eval="eval_real")
            
            if metrics_path_1.exists() and metrics_path_2.exists():
                with open(metrics_path_1, "r") as f:
                    metrics_1 = json.load(f)
                with open(metrics_path_2, "r") as f:
                    metrics_2 = json.load(f)
                
                diff = (metrics_2[metric_name] - metrics_1[metric_name]) * 100
            else:
                raise FileNotFoundError(f"Metrics file not found for {model_name} and {split}")
            
            diff_list.append(diff)
        
        diff_dict[model_name] = diff_list
    
    plot_model_comparison_grouped_categories_bottom(diff_dict)
