import json

import matplotlib.pyplot as plt
import numpy as np

from src.path import figures_dir, human_performance_dir, get_evaluation_metrics_path
from src.config import (
    visonlyqa_real_splits, eval_real_splits_capitalized_full,
    visonlyqa_synthetic_splits, eval_synthetic_splits_capitalized_full,
    convert_model_name, visonlyqa_response_type_dir
)
from src.evaluation.generate_tables import get_random_baseline_performance


model_color_dict = {
    "OpenGVLab/InternVL2-Llama3-76B": "red",
    "claude-3-5-sonnet-20240620": "orange",
    "gpt-4o-2024-08-06": "green",
    "gemini-1.5-pro-002": "blue",
}


def plot_radar_chart(labels: list[str], scores: dict[str, list[float]], title: str, ax: plt.Axes, fontsize: int):
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Create the figure
    for method, values in scores.items():
        values += values[:1]  # Complete the loop
        
        if method == "Random":
            if title == "Eval-Synthetic":
                method = "_nolegend_"
            
            ax.plot(angles, values, label=method, linewidth=2, color="gray", linestyle="--")
        elif method == "Human":
            if title == "Eval-Synthetic":
                method = "_nolegend_"

            ax.plot(angles, values, label=method, linewidth=2)
            ax.fill(angles, values, alpha=0.1)
        else:
            color = model_color_dict[method]
            method = convert_model_name[method]

            if title == "Eval-Synthetic":
                method = "_nolegend_"

            ax.plot(angles, values, label=method, linewidth=2, color=color)
            ax.fill(angles, values, alpha=0.1)
    
    ax.set_title(title, size=fontsize+2, pad=20)
    ax.set_yticks([10, 30, 50, 70, 90])
    ax.set_yticklabels(["10", "30", "50", "70", "90"], color="grey", size=fontsize)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=fontsize-1)


if __name__ == "__main__":
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    splits_dict = {"real": visonlyqa_real_splits, "synthetic": visonlyqa_synthetic_splits}
    
    results_dict: dict[str, dict[str, list[float]]] = {}
    for real_synthetic in ["real", "synthetic"]:
        results_dict[real_synthetic] = {}
        
        # human performance
        human_performance_path = human_performance_dir / f"{real_synthetic}_human_performance.json"
        if human_performance_path.exists():
            with open(human_performance_path, "r") as f:
                human_performance: dict[str, dict[str, float]] = json.load(f)
        else:
            human_performance = None
        
        results_dict[real_synthetic]["Human"] = []
        for split in splits_dict[real_synthetic]:
            key = str((split, "average"))
            if key in human_performance.keys():
                value = human_performance[key]["accuracy"] * 100
            else:
                value = 0
            
            results_dict[real_synthetic]["Human"].append(value)
        
        # model performance
        for model_name in ["OpenGVLab/InternVL2-Llama3-76B", "claude-3-5-sonnet-20240620", "gpt-4o-2024-08-06", "gemini-1.5-pro-002"][::-1]:
            results_dict[real_synthetic][model_name] = []
            
            for split in splits_dict[real_synthetic]:
                metrics_path = get_evaluation_metrics_path(split=split, prompt="no_reasoning", model_name=model_name, train_eval=f"eval_{real_synthetic}")
        
                if metrics_path.exists():
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    metric = metrics["accuracy"] * 100
                else:
                    metric = 0.0
                
                results_dict[real_synthetic][model_name].append(metric)
    
        # Random baseline
        results_dict[real_synthetic]["Random"] = []
        for split in splits_dict[real_synthetic]:
            question_type, options = visonlyqa_response_type_dir[split]
            value = get_random_baseline_performance(question_type=question_type, options_num=len(options), metric="accuracy")
            
            results_dict[real_synthetic]["Random"].append(value * 100)
    
    # Plot the radar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'polar': True}, figsize=(14, 7))
    fontsize = 15

    # Mathematical Reasoning Radar Chart
    plot_radar_chart([s.replace(" - ", "\n") for s in eval_real_splits_capitalized_full],
                     results_dict["real"], "Eval-Real", ax=ax1, fontsize=fontsize)

    # Visual Context Radar Chart
    plot_radar_chart([s.replace(" - ", "\n") for s in eval_synthetic_splits_capitalized_full],
                     results_dict["synthetic"], "Eval-Synthetic", ax=ax2, fontsize=fontsize)

    fig.legend(loc='upper center', fontsize=fontsize-2)

    plt.tight_layout()
    plt.savefig(figures_dir / "accuracy_radar_chart.png")
