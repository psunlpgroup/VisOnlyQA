# run this script after src/evaluation/bootstrap.py
# https://aclanthology.org/W04-3250

import json

from src.path import get_evaluation_model_responses_path, \
    get_paired_bootstrap_path
from src.evaluation.bootstrap import BootstrapTap


class PairedBootstrapTap(BootstrapTap):
    model = None  # not used in this script
    
    model_1: str
    model_2: str


def get_bootstrap_metrics_list(
        bootstrap_result: dict, metric_name: str) -> list:

    bootstrap_result_list: list[dict] = bootstrap_result["y_pred_y_true"]
    
    metrics_list = []
    for d in bootstrap_result_list:
        metrics_list.append(d["metrics"][metric_name])
    
    return metrics_list


def get_paired_bootstrap_p_value(
        bootstrap_result_1: dict, bootstrap_result_2: dict,
        metric_name: str) -> dict:

    metrics_list_1 = get_bootstrap_metrics_list(bootstrap_result_1, metric_name)
    metrics_list_2 = get_bootstrap_metrics_list(bootstrap_result_2, metric_name)
    
    assert len(metrics_list_1) == len(metrics_list_2)
    
    one_is_better = 0
    for metric_1, metric_2 in zip(metrics_list_1, metrics_list_2):
        if metric_1 > metric_2:
            one_is_better += 1
    
    return 1. - one_is_better / len(metrics_list_1),


def main():
    args = PairedBootstrapTap().parse_args()
    
    if "eval" not in args.data:
        raise ValueError("The data argument must contain 'eval'.")
    
    bootstrap_results: list[dict] = []
    for model in [args.model_1, args.model_2]:
        model_responses_path = get_evaluation_model_responses_path(
            split="all", prompt=args.prompt,
            model_name=model, train_eval=args.data
        )
        bootstrap_results_path = model_responses_path.with_suffix(
            ".bootstrap.json"
        )
        
        with open(bootstrap_results_path, "r") as f:
            bootstrap_results.append(json.load(f))
        
    # check if the same seed was used for both models
    if bootstrap_results[0]["seed"] != bootstrap_results[1]["seed"]:
        raise ValueError(
            "The bootstrap seeds for the two models are different."
        )
    
    # paired bootstrap
    paired_bootstrap_result = {}    
    metric_names = list(
        bootstrap_results[0]["y_pred_y_true"][0]["metrics"].keys()
    )
    for metric_name in metric_names:
        paired_bootstrap_result[metric_name] = get_paired_bootstrap_p_value(
            bootstrap_results[0], bootstrap_results[1], metric_name
        )
    
    # save paired bootstrap results
    paired_bootstrap_path = get_paired_bootstrap_path(
        model_1=args.model_1, model_2=args.model_2,
        split="all", prompt=args.prompt, train_eval=args.data
    )
    with open(paired_bootstrap_path, "w") as f:
        json.dump(paired_bootstrap_result, f, indent=4)


if __name__ == "__main__":
    main()
