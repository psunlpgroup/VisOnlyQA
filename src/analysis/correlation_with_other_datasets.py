""" This script calculates the correlation between the performance of VisOnlyQA and other datasets. """


import json

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from src.config import models_list, convert_model_name
from src.path import get_evaluation_metrics_path, openvlm_leaderboard_dir, analysis_dir


convert_model_name_to_openvlm: dict[str, str] = {
    "microsoft/Phi-3.5-vision-instruct": "<a href=\"https://huggingface.co/microsoft/Phi-3.5-vision-instruct\">Phi-3.5-Vision</a>",
    # "xgen-mm-phi3-interleave-r-v1.5": "<a href=\"https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5\">XGen-MM-Instruct-Interleave-v1.5</a>",
    #
    "llava_next_llama3": "<a href=\"https://llava-vl.github.io/blog/2024-01-30-llava-next/\">LLaVA-Next-Llama3</a>",
    "llava_next_yi_34b": "<a href=\"https://llava-vl.github.io/blog/2024-01-30-llava-next/\">LLaVA-Next-Yi-34B</a>",
    #
    "OpenGVLab/InternVL2-4B": "<a href=\"https://huggingface.co/OpenGVLab/InternVL2-4B\">InternVL2-4B</a>",
    "OpenGVLab/InternVL2-8B": "<a href=\"https://huggingface.co/OpenGVLab/InternVL2-8B\">InternVL2-8B</a>",
    "OpenGVLab/InternVL2-26B": "<a href=\"https://huggingface.co/OpenGVLab/InternVL2-26B\">InternVL2-26B</a>",
    "OpenGVLab/InternVL2-40B": "<a href=\"https://huggingface.co/OpenGVLab/InternVL2-40B\">InternVL2-40B</a>",
    "OpenGVLab/InternVL2-Llama3-76B": "<a href=\"https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B\">InternVL2-Llama3-76B</a>",
    #
    "claude-3-5-sonnet-20240620": "<a href=\"https://docs.anthropic.com/claude/docs/vision\">Claude3.5-Sonnet</a>",
    "gpt-4o-mini-2024-07-18": "<a href=\"https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/\">GPT-4o-mini (0718, detail-high)</a>",
    "gpt-4o-2024-08-06": "<a href=\"https://openai.com/index/hello-gpt-4o/\">GPT-4o (0806, detail-high)</a>",
    # "gemini-1.5-pro-002": "<a href=\"https://deepmind.google/technologies/gemini/\">Gemini-1.5-Pro</a>",
}


if __name__ == "__main__":
    performance_dict: dict[str, list[float]] = {}
    
    models_list = [m for m in models_list if m in convert_model_name_to_openvlm.keys()]
    
    # read visonlyqa results
    visonlyqa_accuracy: list[float] = []
    for model_name in models_list:
        with open(get_evaluation_metrics_path(split="all", prompt="no_reasoning", model_name=model_name, train_eval="eval_real")) as f:
            metrics = json.load(f)
            visonlyqa_accuracy.append(metrics["accuracy"] * 100)
    performance_dict["VisOnlyQA"] = visonlyqa_accuracy
    
    # read openvlm results
    with open(openvlm_leaderboard_dir / "openvlm.json") as f:
        openvlm_leaderboard = json.load(f)["value"]
    openvlm_data: list[list] = openvlm_leaderboard["data"]
    
    header: list[str] = openvlm_leaderboard["headers"]
    model_name_idx = header.index("Method")
    
    dataset_names_list = ["MMMU_VAL","MMVet","OCRBench","MathVista","HallusionBench","SEEDBench2_Plus","AI2D","MMBench_V11","CCBench","MME","SEEDBench_IMG"]  # "MMStar", "LLaVABench" # "RealWorldQA","POPE","ScienceQA_TEST",]  # ,"MMM-Bench_VAL","BLINK"]
    for dataset_name in dataset_names_list:
        dataset_idx = header.index(dataset_name)
        
        openvlm_accuracy: list[float] = []
        for model_name in models_list:
            openvlm_model_name = convert_model_name_to_openvlm[model_name]
            
            # from openvlm_data, find a row whose model_name_idx is equal to openvlm_model_name
            # raise error if not found
            found = False
            for row in openvlm_data:
                if row[model_name_idx] == openvlm_model_name:
                    found = True
                    break
            if not found:
                raise Exception(f"Model {model_name} not found in OpenVLM Leaderboard")
            
            openvlm_accuracy.append(row[dataset_idx])
        
        performance_dict[dataset_name] = openvlm_accuracy
    
    correlation_results_dir = analysis_dir / "correlation_results"
    correlation_results_dir.mkdir(parents=True, exist_ok=True)
    
    # correlation matrix between datasets
    dataset_names_list = ["VisOnlyQA"] + dataset_names_list
    correlation_matrix: dict[str, list[list[float]]] = {}
    
    for cor_name in ["pearson", "spearman"]:
        correlation_matrix[cor_name] = []
        for dataset_name_1 in dataset_names_list:
            correlation_row: list[float] = []
            for dataset_name_2 in dataset_names_list:
                if cor_name == "pearson":
                    cor, _ = scipy.stats.pearsonr(performance_dict[dataset_name_1], performance_dict[dataset_name_2])
                else:
                    cor, _ = scipy.stats.spearmanr(performance_dict[dataset_name_1], performance_dict[dataset_name_2])
                correlation_row.append(cor)
            correlation_matrix[cor_name].append(correlation_row)
    
    with open(correlation_results_dir / "correlation.json", "w") as f:
        json.dump(correlation_matrix, f, indent=4)
    
    # draw a heatmap for the correlation matrix
    for cor_name in ["pearson", "spearman"]:
        cor_mat = np.array(correlation_matrix[cor_name])
        
        plt.figure(figsize=(8, 8))
        
        colors = ['#FFF5E1', '#FEC96F', '#F98F52', '#E64B35', '#990000']
        cmap = ListedColormap(colors)
        bounds = np.arange(0.6, 1.1, 0.1)
        norm = BoundaryNorm(bounds, cmap.N)
        
        heatmap = plt.imshow(cor_mat, cmap=cmap, norm=norm) # cmap='YlOrRd', interpolation='nearest', vmin=0.6, vmax=1)  

        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')

        plt.xticks(np.arange(len(dataset_names_list)), dataset_names_list, rotation=90)
        plt.yticks(np.arange(len(dataset_names_list)), dataset_names_list)

        for i in range(len(cor_mat)):
            for j in range(len(cor_mat)):
                plt.text(j, i, f'{cor_mat[i, j]:.2f}', ha='center', va='center', color='black')

        plt.subplots_adjust(top=0.8)
        plt.savefig(correlation_results_dir / f"correlation_heatmap_{cor_name}.png")

    # latex table of performance
    table = []
    first_row = ["Model"] + dataset_names_list + ["VisOnlyQA"]
    table.append(" & ".join(first_row) + " \\\\")
    for idx, model_name in enumerate(models_list):
        row = [convert_model_name[model_name]]
        for dataset_name in dataset_names_list + ["VisOnlyQA"]:
            accuracy = performance_dict[dataset_name][idx]
            row.append(f"{accuracy:.1f}")
        row_latex = " & ".join(row) + " \\\\"
        table.append(row_latex)
    
    with open(correlation_results_dir / "performance_table.txt", "w") as f:
        f.write("\n".join(table))
