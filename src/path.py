from pathlib import Path
from typing import Literal

# VisOnlyQA
dataset_dir = Path("./dataset")
train_dataset_dir = dataset_dir / "train"
val_dataset_dir = dataset_dir / "val"
test_dataset_dir = dataset_dir / "test"

# dataset creation
source_dataset_download_dir = Path("../datasets")

intermediate_dir = Path("./intermediate")
test_intermediate_dir = intermediate_dir / "test"
test_intermediate_data_dir = test_intermediate_dir / "data"
test_annotations_dir = test_intermediate_dir / "annotations"

train_intermediate_dir = intermediate_dir / "train"
train_intermediate_data_dir = train_intermediate_dir / "data"

val_intermediate_dir = intermediate_dir / "val"
val_intermediate_data_dir = val_intermediate_dir / "data"

alphageometry_intermediate_dir = intermediate_dir / "alphageometry"

test_image_path_dir = test_intermediate_dir / "image_path"
test_human_performance_annotation_dir = test_intermediate_dir / "human_performance_annotation"

config_dir = Path("./config")

llama_factory_dir = Path("../LLaMA-Factory")
llama_factory_data_info_path = llama_factory_dir / "data/dataset_info.json"

# stats
dataset_stats_dir = Path("./dataset_stats")
eval_real_dataset_stats_path = dataset_stats_dir / "visonlyqa-eval-real_dataset_stats.json"
eval_synthetic_dataset_stats_path = dataset_stats_dir / "visonlyqa-eval-synthetic_dataset_stats.json"
eval_synthetic_with_text_dataset_stats_path = dataset_stats_dir / "visonlyqa-eval-synthetic_with_text_dataset_stats.json"
train_dataset_stats_path = dataset_stats_dir / "visonlyqa-train_dataset_stats.json"

# evaluation
results_dir = Path("results")
model_responses_dir = results_dir / "model_responses"
evaluation_metrics_dir = results_dir / "evaluation_metrics"
tables_dir = results_dir / "tables"
figures_dir = results_dir / "figures"
human_performance_dir = results_dir / "human_performance"

# analysis
analysis_dir = results_dir / "analysis"
openvlm_leaderboard_dir = analysis_dir / "openvlm_leaderboard"

# VLMEvalKit
vlmevalkit_models_list_path = Path("config/vlmevalkit_models_list.txt")

# InternVL experiments
# This experiments did not work. This is not included in the paper.
new_internvl_models_dir = Path("new_internvl_models")


def get_evaluation_model_responses_path(split: str, prompt: Literal["reasoning", "no_reasoning"], model_name: str, train_eval: Literal["train", "eval_real", "eval_synthetic"]) -> Path:
    from src.utils.utils import get_short_model_name
    
    path = model_responses_dir / train_eval / split / f"prompt={prompt}" / f"{get_short_model_name(model_name)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return path


def get_evaluation_metrics_path(split: str, prompt: Literal["reasoning", "no_reasoning"], model_name: str, train_eval: Literal["train", "eval_real", "eval_synthetic"]) -> Path:
    from src.utils.utils import get_short_model_name
    
    path = evaluation_metrics_dir / train_eval / split / f"prompt={prompt}" / f"{get_short_model_name(model_name)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    return path
