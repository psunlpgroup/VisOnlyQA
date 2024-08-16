from typing import Union, Literal


# dataset
huggingface_eval_real_data_name = "VisOnlyQA_Eval_Real"
huggingface_eval_synthetic_data_name = "VisOnlyQA_Eval_Synthetic"
huggingface_eval_synthetic_with_text_data_name = "VisOnlyQA_Eval_Synthetic_with_Text"
huggingface_train_data_name = "VisOnlyQA_Train"

# evaluation dataset
visonlyqa_real_splits = [
    "geometry__triangle", "geometry__quadrilateral", "geometry__length", "geometry__angle", "geometry__area", "geometry__diameter_radius", 
    "chemistry__shape_single", "chemistry__shape_multi",
    "charts__extraction", "charts__intersection",
]

visonlyqa_synthetic_splits = [
    "syntheticgeometry__triangle", "syntheticgeometry__quadrilateral", "syntheticgeometry__length", "syntheticgeometry__angle", "syntheticgeometry__area",
    "3d__size", "3d__angle",
]

visonlyqa_synthetic_with_text_splits = [f"text_{s}" for s in visonlyqa_synthetic_splits]  # this is not used in the paper

visonlyqa_response_type_dir: dict[str, tuple[str, list[str]]] = {
    #
    # Real
    #
    "geometry__triangle": ("single_answer", ["True", "False"]),
    "geometry__quadrilateral": ("single_answer", ["True", "False"]),
    "geometry__length": ("single_answer", ["a", "b", "c", "d", "e"]),
    "geometry__angle": ("single_answer", ["a", "b", "c", "d", "e"]),
    "geometry__area": ("single_answer", ["a", "b", "c", "d", "e"]),
    "geometry__diameter_radius": ("single_answer", ["True", "False"]),
    #
    "chemistry__shape_single": ("single_answer", ["True", "False"]),
    "chemistry__shape_multi": ("multiple_answers", ["a", "b", "c", "d"]),
    #
    "charts__extraction": ("single_answer", ["a", "b", "c", "d", "e"]),
    "charts__intersection": ("single_answer", ["True", "False"]),
    #
    # Synthetic
    #
    "3d__size": ("single_answer", ["a", "b", "c"]),
    "3d__angle": ("single_answer", ["a", "b", "c", "d", "e"]),
    #
    "syntheticgeometry__triangle": ("single_answer", ["True", "False"]),
    "syntheticgeometry__quadrilateral": ("single_answer", ["True", "False"]),
    "syntheticgeometry__length": ("single_answer", ["a", "b", "c", "d", "e"]),
    "syntheticgeometry__angle": ("single_answer", ["a", "b", "c", "d", "e"]),
    "syntheticgeometry__area": ("single_answer", ["a", "b", "c", "d", "e"]),
    #
    "text_3d__size": ("single_answer", ["a", "b", "c"]),
    "text_3d__angle": ("single_answer", ["a", "b", "c", "d", "e"]),
    #
    "text_syntheticgeometry__triangle": ("single_answer", ["True", "False"]),
    "text_syntheticgeometry__quadrilateral": ("single_answer", ["True", "False"]),
    "text_syntheticgeometry__length": ("single_answer", ["a", "b", "c", "d", "e"]),
    "text_syntheticgeometry__angle": ("single_answer", ["a", "b", "c", "d", "e"]),
    "text_syntheticgeometry__area": ("single_answer", ["a", "b", "c", "d", "e"]),
}

# train dataset
finetuning_splits_dict: dict[Literal["finetuning_splits", "corresponding_real_splits", "finetuning_text_splits"], list[Union[str, None]]] = {
    "finetuning_splits": visonlyqa_synthetic_splits,
    "corresponding_real_splits": [
        "geometry__triangle", "geometry__quadrilateral", "geometry__length", "geometry__angle", "geometry__area",
        None, None,
    ],
    "finetuning_text_splits": visonlyqa_synthetic_with_text_splits,
}
train_data_splits = finetuning_splits_dict["finetuning_splits"]
train_data_text_splits = visonlyqa_synthetic_with_text_splits

# models
open_models_list = [
    "microsoft/Phi-3.5-vision-instruct",
    "llava_next_llama3", "llava_next_yi_34b",
    "Llama-3.2-11B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct",
    "molmo-7B-D-0924", "molmo-72B-0924",
    "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct", "Qwen/Qwen2-VL-72B-Instruct",
    "OpenGVLab/InternVL2-4B", "OpenGVLab/InternVL2-8B", "OpenGVLab/InternVL2-26B", "OpenGVLab/InternVL2-40B", "OpenGVLab/InternVL2-Llama3-76B",
]
proprietary_models_list = ["claude-3-5-sonnet-20240620", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gemini-1.5-flash-002", "gemini-1.5-pro-002"]
models_list = open_models_list + proprietary_models_list

convert_model_name = {
    "microsoft/Phi-3.5-vision-instruct": "Phi-3.5-vision",
    "llava_next_llama3": "LLaVA-Next 8B",
    "llava_next_yi_34b": "LLaVA-Next 34B",
    #
    "Qwen/Qwen2-VL-2B-Instruct": "Qwen2-VL-2B",
    "Qwen/Qwen2-VL-7B-Instruct": "Qwen2-VL-7B",
    "Qwen/Qwen2-VL-72B-Instruct": "Qwen2-VL-72B",
    #
    "molmo-7B-D-0924": "MolMo 7B-D",
    "molmo-72B-0924": "MolMo 72B",
    #
    "Llama-3.2-11B-Vision-Instruct": "Llama 3.2 11B",
    "Llama-3.2-90B-Vision-Instruct": "Llama 3.2 90B",
    #
    "OpenGVLab/InternVL2-4B": "InternVL2-4B",
    "OpenGVLab/InternVL2-8B": "InternVL2-8B",
    "OpenGVLab/InternVL2-26B": "InternVL2-26B",
    "OpenGVLab/InternVL2-40B": "InternVL2-40B",
    "OpenGVLab/InternVL2-Llama3-76B": "InternVL2-76B",
    #
    "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
    #
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gpt-4o-2024-08-06": "GPT-4o",
    #
    "gemini-1.5-flash-002": "Gemini 1.5 Flash",
    "gemini-1.5-pro-002": "Gemini 1.5 Pro",
}

model_row_colors = {  # for the tables in the paper
    "microsoft/Phi-3.5-vision-instruct": "cyan",
    "llava_next_yi_34b": "Orange",
    "Qwen/Qwen2-VL-72B-Instruct": "RoyalPurple",
    "Llama-3.2-90B-Vision-Instruct": "Blue",
    "molmo-72B-0924": "Lavender",
    "OpenGVLab/InternVL2-Llama3-76B": "RedOrange",
    "claude-3-5-sonnet-20240620": "YellowOrange",
    "gpt-4o-2024-08-06": "Green",
    "gemini-1.5-pro-002": "BlueViolet",
}

# for these models we use for fine-tuning, we implement model specific code
open_models_with_specific_code_list = ["InternVL", "Phi-3.5", "Qwen2-VL"]

finetuning_base_models_list = [
    "microsoft/Phi-3.5-vision-instruct",
    "OpenGVLab/InternVL2-4B", "OpenGVLab/InternVL2-8B",
    "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"
]
base_model_to_finetuned_model_dict: dict[str, dict[str, str]] = {
    "OpenGVLab/InternVL2-4B": {
        "syntheticgeometry__triangle": "finetuning_results/internvl_finetuning_log/InternVL2-4B_syntheticgeometry__triangle_20241106_213204",
        "syntheticgeometry__quadrilateral": "finetuning_results/internvl_finetuning_log/InternVL2-4B_syntheticgeometry__quadrilateral_20241106_222740",
        "syntheticgeometry__length": "finetuning_results/internvl_finetuning_log/InternVL2-4B_syntheticgeometry__length_20241107_011728",
        "syntheticgeometry__angle": "finetuning_results/internvl_finetuning_log/InternVL2-4B_syntheticgeometry__angle_20241106_232458",
        "syntheticgeometry__area": "finetuning_results/internvl_finetuning_log/InternVL2-4B_syntheticgeometry__area_20241107_002108",
        "3d__size": "finetuning_results/internvl_finetuning_log/InternVL2-4B_3d__size_20241106_193554",
        "3d__angle": "finetuning_results/internvl_finetuning_log/InternVL2-4B_3d__angle_20241106_203320",
        "text_syntheticgeometry__angle": "finetuning_results/internvl_finetuning_log/InternVL2-4B_text_syntheticgeometry__angle_20241126_015556",
        "text_syntheticgeometry__area": "finetuning_results/internvl_finetuning_log/InternVL2-4B_text_syntheticgeometry__area_20241126_030459",
        "text_syntheticgeometry__length": "finetuning_results/internvl_finetuning_log/InternVL2-4B_text_syntheticgeometry__length_20241126_041211",
        "text_syntheticgeometry__quadrilateral": "finetuning_results/internvl_finetuning_log/InternVL2-4B_text_syntheticgeometry__quadrilateral_20241126_005040",
        "text_syntheticgeometry__triangle": "finetuning_results/internvl_finetuning_log/InternVL2-4B_text_syntheticgeometry__triangle_20241125_234554",
        "text_3d__size": "finetuning_results/internvl_finetuning_log/InternVL2-4B_text_3d__size_20241125_212829",
        "text_3d__angle": "finetuning_results/internvl_finetuning_log/InternVL2-4B_text_3d__angle_20241125_223732",
    },
    "OpenGVLab/InternVL2-8B": {
        "syntheticgeometry__triangle": "finetuning_results/internvl_finetuning_log/InternVL2-8B_syntheticgeometry__triangle_20241106_200045",
        "syntheticgeometry__quadrilateral": "finetuning_results/internvl_finetuning_log/InternVL2-8B_syntheticgeometry__quadrilateral_20241106_213050",
        "syntheticgeometry__length": "finetuning_results/internvl_finetuning_log/InternVL2-8B_syntheticgeometry__length_20241107_020956",
        "syntheticgeometry__angle": "finetuning_results/internvl_finetuning_log/InternVL2-8B_syntheticgeometry__angle_20241106_230117",
        "syntheticgeometry__area": "finetuning_results/internvl_finetuning_log/InternVL2-8B_syntheticgeometry__area_20241107_003601",
        "3d__size": "finetuning_results/internvl_finetuning_log/InternVL2-8B_3d__size_20241106_165308",
        "3d__angle": "finetuning_results/internvl_finetuning_log/InternVL2-8B_3d__angle_20241106_182532",
        "text_syntheticgeometry__angle": "finetuning_results/internvl_finetuning_log/InternVL2-8B_text_syntheticgeometry__angle_20241125_023740",
        "text_syntheticgeometry__area": "finetuning_results/internvl_finetuning_log/InternVL2-8B_text_syntheticgeometry__area_20241125_042312",
        "text_syntheticgeometry__length": "finetuning_results/internvl_finetuning_log/InternVL2-8B_text_syntheticgeometry__length_20241125_060917",
        "text_syntheticgeometry__quadrilateral": "finetuning_results/internvl_finetuning_log/InternVL2-8B_text_syntheticgeometry__quadrilateral_20241125_005431",
        "text_syntheticgeometry__triangle": "finetuning_results/internvl_finetuning_log/InternVL2-8B_text_syntheticgeometry__triangle_20241124_231134",
        "text_3d__size": "finetuning_results/internvl_finetuning_log/InternVL2-8B_text_3d__size_20241124_193710",
        "text_3d__angle": "finetuning_results/internvl_finetuning_log/InternVL2-8B_text_3d__angle_20241124_212445",
    },
    "Qwen/Qwen2-VL-2B-Instruct": {
        "3d__angle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_3d__angle",
        "3d__size": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_3d__size",
        "syntheticgeometry__angle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_syntheticgeometry__angle",
        "syntheticgeometry__area": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_syntheticgeometry__area",
        "syntheticgeometry__length": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_syntheticgeometry__length",
        "syntheticgeometry__quadrilateral": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_syntheticgeometry__quadrilateral",
        "syntheticgeometry__triangle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_syntheticgeometry__triangle",
        "text_3d__angle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_text_3d__angle",
        "text_3d__size": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_text_3d__size",
        "text_syntheticgeometry__angle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_text_syntheticgeometry__angle",
        "text_syntheticgeometry__area": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_text_syntheticgeometry__area",
        "text_syntheticgeometry__length": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_text_syntheticgeometry__length",
        "text_syntheticgeometry__quadrilateral": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_text_syntheticgeometry__quadrilateral",
        "text_syntheticgeometry__triangle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-2B-Instruct_finetuned_text_syntheticgeometry__triangle",
    },
    "Qwen/Qwen2-VL-7B-Instruct": {
        "3d__angle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_3d__angle",
        "3d__size": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_3d__size",
        "syntheticgeometry__angle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_syntheticgeometry__angle",
        "syntheticgeometry__area": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_syntheticgeometry__area",
        "syntheticgeometry__length": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_syntheticgeometry__length",
        "syntheticgeometry__quadrilateral": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_syntheticgeometry__quadrilateral",
        "syntheticgeometry__triangle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_syntheticgeometry__triangle",
        "text_3d__angle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_text_3d__angle",
        "text_3d__size": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_text_3d__size",
        "text_syntheticgeometry__angle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_text_syntheticgeometry__angle",
        "text_syntheticgeometry__area": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_text_syntheticgeometry__area",
        "text_syntheticgeometry__length": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_text_syntheticgeometry__length",
        "text_syntheticgeometry__quadrilateral": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_text_syntheticgeometry__quadrilateral",
        "text_syntheticgeometry__triangle": "finetuning_results/qwen2vl_finetuning_log/Qwen2-VL-7B-Instruct_finetuned_text_syntheticgeometry__triangle",
    },
    "microsoft/Phi-3.5-vision-instruct": {
        "3d__angle": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_3d__angle_2024-11-26_02-31-59",
        "3d__size": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_3d__size_2024-11-25_23-36-05",
        "syntheticgeometry__angle": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_syntheticgeometry__angle_2024-11-05_14-58-51",
        "syntheticgeometry__area": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_syntheticgeometry__area_2024-11-25_19-39-18",
        "syntheticgeometry__triangle": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_syntheticgeometry__triangle_2024-11-27_00-39-36",
        "syntheticgeometry__quadrilateral": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_syntheticgeometry__quadrilateral_2024-11-25_13-31-01",
        "syntheticgeometry__length": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_syntheticgeometry__length_2024-11-25_16-35-41",
        "text_3d__size": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_text_3d__size_2024-11-26_19-57-29",
        "text_3d__angle": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_text_3d__angle_2024-11-26_22-48-57",
        "text_syntheticgeometry__angle": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_text_syntheticgeometry__angle_2024-11-26_16-28-27",
        "text_syntheticgeometry__area": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_text_syntheticgeometry__area_2024-11-27_12-37-00",
        "text_syntheticgeometry__quadrilateral": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_text_syntheticgeometry__quadrilateral_2024-11-26_08-29-45",
        "text_syntheticgeometry__triangle": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_text_syntheticgeometry__triangle_2024-11-26_05-06-20",
        "text_syntheticgeometry__length": "finetuning_results/phi35v_finetuning_log/Phi-3.5-vision-instruct_text_syntheticgeometry__length_2024-11-26_11-53-56",
    }
}


# image tokens in prompt
model_image_tokens_dict: dict[str, str] = {
    "InternVL": "<image>",
    "Phi-3.5": "<|image_1|>"
}


def convert_split_name(split_name: str, full=False) -> str:
    category_name, split_name = split_name.split("__")
    
    if split_name == "diameter_radius":
        split_str= "Diameter"
    elif split_name == "shape_single":
        split_str= "Shape (s)"
    elif split_name == "shape_multi":
        split_str= "Shape (m)"
    else:
        split_str = split_name.replace("_", " ").title()
    
    if full:
        split_str = f"{category_name.title()} - {split_str}"
    
    return split_str

eval_real_splits_capitalized = [convert_split_name(split) for split in visonlyqa_real_splits]
eval_synthetic_splits_capitalized = [convert_split_name(split) for split in visonlyqa_synthetic_splits]
eval_synthetic_with_text_splits_capitalized = [convert_split_name(split) for split in visonlyqa_synthetic_with_text_splits]
finetuning_splits_capitalized = [convert_split_name(split) for split in finetuning_splits_dict["finetuning_splits"]]

eval_real_splits_capitalized_full = [convert_split_name(split, full=True) for split in visonlyqa_real_splits]
eval_synthetic_splits_capitalized_full = [convert_split_name(split, full=True) for split in visonlyqa_synthetic_splits]
eval_synthetic_with_text_splits_capitalized_full = [convert_split_name(split, full=True) for split in visonlyqa_synthetic_with_text_splits]
finetuning_splits_capitalized_full = [convert_split_name(split, full=True) for split in finetuning_splits_dict["finetuning_splits"]]
