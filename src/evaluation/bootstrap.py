import json
from tqdm import tqdm
from typing import TypedDict

import datasets
import numpy as np
from pathlib import Path

from src.config import visonlyqa_real_splits, visonlyqa_synthetic_splits, train_data_splits
from src.path import get_evaluation_model_responses_path, get_evaluation_metrics_path
from src.evaluation.evaluation import EvaluationTap
from src.evaluation.metrics import get_metrics
from src.utils import get_hf_dataset_name


class BootstrapTap(EvaluationTap):
    num_bootstrap_samples: int = 1000


class BootstrapResult(TypedDict):
    std: dict[str, float]  # standard deviation of metrics


def get_bootstrap_result(y_true: list, y_pred: list, 
        num_bootstrap_samples: int = 1000,
        seed: int = 68) -> tuple[BootstrapResult, list[dict[str, list]]]:
    """Calculate bootstrap results for a given list of true and pred values.
    
    Args:
        y_true: list of true values
        y_pred: list of predicted values
        num_bootstrap_samples: number of bootstrap samples
        seed: random seed for reproducibility
    
    Returns:
        BootstrapResult: """

    bootstrap_metrics_dict_of_lists: dict[str, list] = {}
    bootstrap_y_pred_y_true: list[dict[str, list]] = []
    for sample_seed in tqdm(range(seed, seed + num_bootstrap_samples)):
        bootstrap_index = np.random.RandomState(sample_seed).choice(
            len(y_true), len(y_true), replace=True
        )
        
        bootstrap_y_true = [y_true[i] for i in bootstrap_index]
        bootstrap_y_pred = [y_pred[i] for i in bootstrap_index]
        
        metrics = get_metrics(bootstrap_y_true, bootstrap_y_pred)
        
        # store metrics in dict of lists
        for metric_name, metric_value in metrics.items():
            bootstrap_metrics_dict_of_lists.setdefault(
                metric_name, []).append(metric_value)
        
        # store y_true and y_pred
        bootstrap_y_pred_y_true.append(
            {
                "y_true": bootstrap_y_true, "y_pred": bootstrap_y_pred,
                "metrics": metrics,
            }
        )
    
    ###
    # bootstrap results
    bootstrap_result = BootstrapResult()
    
    # calculate standard deviation
    bootstrap_result["std"] = {}
    for metric_name, metric_values in bootstrap_metrics_dict_of_lists.items():
        bootstrap_result["std"][metric_name] = np.std(metric_values)
    
    return bootstrap_result, bootstrap_y_pred_y_true


def save_bootstrap_y_pred_y_true(
        bootstrap_y_pred_y_true: list[dict[str, list]],
        model_response_path: Path, seed: int
    ) -> None:

    bootstrap_path = model_response_path.with_suffix(".bootstrap.json")
    
    # when we perform a paired bootstrap test, we need to know the seed to
    # check if the same seed was used for both models
    save_dict = {"y_pred_y_true": bootstrap_y_pred_y_true, "seed": seed}
    
    # save y_true and y_pred
    with open(bootstrap_path, "w") as f:
        json.dump(save_dict, f)


def add_bootstrap_result_to_evaluation_metrics(
        args: BootstrapTap, split: str,
        bootstrap_result: BootstrapResult) -> None:
    
    evaluation_metrics_path = get_evaluation_metrics_path(
        split=split, prompt=args.prompt,
        model_name=args.model, train_eval=args.data
    )
    
    # load evaluation metrics
    with open(evaluation_metrics_path, "r") as f:
        evaluation_metrics = json.load(f)
    
    # add bootstrap result to the evaluation metrics
    evaluation_metrics["bootstrap"] = bootstrap_result
    
    # save evaluation metrics
    with open(evaluation_metrics_path, "w") as f:
        json.dump(evaluation_metrics, f, indent=4)


def main():
    args = BootstrapTap().parse_args()
    
    repository_name = get_hf_dataset_name(args.data)
    
    splits_list = {
        "train": ["train_all_first_100"] + [f"{split}_50" for split in train_data_splits],
        "eval_real": visonlyqa_real_splits,
        "eval_synthetic": visonlyqa_synthetic_splits,
    }[args.data]
    
    if "finetuning_results" in args.model:
        splits_list = [split for split in splits_list if split in args.model]
    
    all_y_pred = []
    all_y_true = []
    for split in splits_list:
        print(f"Split: {split}")
        dataset = datasets.load_dataset(repository_name, split=split)
        y_true = dataset["answer"]
        
        # load y_pred
        model_resposnes_path = get_evaluation_model_responses_path(
            split=split, prompt=args.prompt,
            model_name=args.model, train_eval=args.data
        )
        postprocessed_output_path = model_resposnes_path.with_suffix(
            ".postprocessed.json"
        )
        with open(postprocessed_output_path, "r") as f:
            y_pred = json.load(f)
        
        # bootstrap
        bootstrap_result, bootstrap_y_pred_y_true = get_bootstrap_result(
            y_true, y_pred, 
            num_bootstrap_samples=args.num_bootstrap_samples, seed=args.seed
        )
        
        # add bootstrap result to the evaluation metrics
        add_bootstrap_result_to_evaluation_metrics(
            args=args, split=split, bootstrap_result=bootstrap_result
        )
        
        # save bootstrap result
        # we can use this later for a paired bootstrap test
        save_bootstrap_y_pred_y_true(
            bootstrap_y_pred_y_true, model_resposnes_path, seed=args.seed
        )
        
        # all y_pred and y_true
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
    
    if "eval" in args.data:
        # boostrap result for all splits
        all_bootstrap_result, bootstrap_y_pred_y_true = get_bootstrap_result(
            all_y_true, all_y_pred, 
            num_bootstrap_samples=args.num_bootstrap_samples, seed=args.seed
        )
        
        # add bootstrap result to the evaluation metrics
        add_bootstrap_result_to_evaluation_metrics(
            args=args, split="all", bootstrap_result=all_bootstrap_result
        )

        # save bootstrap result
        # we can use this later for a paired bootstrap test
        model_resposnes_path = get_evaluation_model_responses_path(
            split="all", prompt=args.prompt,
            model_name=args.model, train_eval=args.data
        )
        model_resposnes_path.parent.mkdir(parents=True, exist_ok=True)
        save_bootstrap_y_pred_y_true(
            bootstrap_y_pred_y_true, model_resposnes_path, seed=args.seed
        )


if __name__ == "__main__":
    main()
