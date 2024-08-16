""" Generate a table for examples in the paper. We will include randomly selected examples from the dataset and model responses. """

import random
import shutil
import json

import datasets

from src.config import visonlyqa_real_splits, visonlyqa_synthetic_splits, convert_model_name
from src.path import tables_dir, intermediate_dir, test_dataset_dir, get_evaluation_model_responses_path
from src.utils import get_hf_dataset_name

num_sample_for_each_split = 3
table_figure_dir_in_latex = "figures/data-examples"

examples_table_path = tables_dir / "dataset_examples_in_paper.tex"
examples_image_dir = intermediate_dir / "images_for_examples_in_paper"

example_table_prefix = """%
%
%
\\begin{table*}[t]
    \\centering
    \\scriptsize
    %
    \\begin{tabular}{cL{.5\linewidth}M{.1\linewidth}}
    \\toprule
        Image & \\multicolumn{1}{c}{Question} & Gold Answer \\\\
    \\midrule
"""

example_table_suffix = """\\bottomrule
    \\end{tabular}
    %
\\vskip 1em
%
%
%
"""

response_example_models_list = [
    "OpenGVLab/InternVL2-4B", "OpenGVLab/InternVL2-8B", "OpenGVLab/InternVL2-26B",
    "OpenGVLab/InternVL2-Llama3-76B",
    "claude-3-5-sonnet-20240620", "gpt-4o-2024-08-06", "gemini-1.5-pro-002"
]

response_example_table_prefix = """%
    \\begin{tabular}{M{.1\linewidth}L{.65\linewidth}M{.04\linewidth}M{.07\linewidth}M{.04\linewidth}}
    \\toprule
    Model & \\multicolumn{1}{c}{Answer w/ CoT} &  & Answer w/o CoT & \\\\
    \\midrule
"""

response_example_table_suffix = """\\bottomrule
    \\end{{tabular}}
    %
    \\caption{{Examples of dataset and model responses for \\texttt{{{split}}} ({index})}} \\label{{tab:examples_{split}_{index}}}%
\\end{{table*}}
%
%
%
\\clearpage
"""

replace_list = [["△", "$\\ensuremath{\\triangle}$"], ["∠", "$\\ensuremath{\\angle}$"], ["\\", "\\textbackslash "], ["₃", "$_3$"]]
add_backslash_list = ["_", "&", "%", "$", "#", "^", "{", "}", "~"]

def update_text_for_latex(text: str) -> str:
    for replace_pair in replace_list:
        text = text.replace(replace_pair[0], replace_pair[1])

    for symbol in add_backslash_list:
        text = text.replace(symbol, f"\\{symbol}")
    
    return text


if __name__ == "__main__":
    examples_image_dir.mkdir(parents=True, exist_ok=True)
    
    table = ""
    for real_synthetic in ["real", "synthetic"]:
        repository_name = get_hf_dataset_name(f"eval_{real_synthetic}")
        
        splits_list = {
            "real": visonlyqa_real_splits,
            "synthetic": visonlyqa_synthetic_splits,
        }[real_synthetic]
        
        for split in splits_list:
            print(f"Split: {real_synthetic} - {split}")
            
            dataset = datasets.load_dataset(repository_name, split=split)
            
            random_indices_list = random.Random(f"{real_synthetic}_{split}").sample(range(len(dataset)), num_sample_for_each_split)
            
            # create latex table
            for enumerate_index, random_index in enumerate(random_indices_list):
                table += example_table_prefix
            
                # response
                d = dataset[random_index]
                
                image_path: str = d["image_path"]
                image_name = image_path.split("/")[-1]
                image_path_in_latex = f"{table_figure_dir_in_latex}/{image_name}"
                
                row = [
                    f"\\includegraphics[width=0.3\\linewidth]{{{image_path_in_latex}}}",
                    update_text_for_latex(d["question"]),
                    update_text_for_latex(d["answer"]),
                ]
                
                table += " & ".join(row) + " \\\\\n" + example_table_suffix
                
                # copy image
                target_image_path = examples_image_dir / image_name
                shutil.copy(test_dataset_dir / d["image_path"], target_image_path)
                
                # model responses
                table += response_example_table_prefix
                for model_name in response_example_models_list:
                    row: list[str] = []
                    
                    model_name_str = convert_model_name[model_name]
                    row.append(model_name_str)
                    
                    for prompt_type in ["reasoning", "no_reasoning"]:
                        if real_synthetic == "synthetic" and prompt_type == "reasoning":
                            row.extend(["\\multicolumn{1}{c}{--}", "--"])
                            continue
                        
                        response_path = get_evaluation_model_responses_path(split, prompt_type, model_name, f"eval_{real_synthetic}")
                        responses_list: list[dict] = []
                        with open(response_path, "r") as f:
                            for line in f:
                                responses_list.append(json.loads(line))
                        
                        selected_response = responses_list[random_index]
                        
                        # emoji checkbox or cross mark for latex
                        correct_wrong = "\\cmark" if selected_response["y_pred"] == selected_response["y_true"] else "\\xmark"
                        
                        response = update_text_for_latex(selected_response["response"])
                        
                        row.extend([response, correct_wrong])
                    
                    table += " & ".join(row) + " \\\\\n"
                    if model_name != response_example_models_list[-1]:
                        table += "\\midrule\n"
                
                # capitalize
                split_str = split.replace("__", "-").replace("_", "-").title()
                table += response_example_table_suffix.format(split=split_str, index=enumerate_index+1)
    
    with open(examples_table_path, "w") as f:
        f.write(table)
