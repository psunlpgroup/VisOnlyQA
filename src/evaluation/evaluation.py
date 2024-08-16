import json
from tqdm import tqdm
from typing import Literal

import torch
from tap import Tap
import datasets

from src.config import visonlyqa_real_splits, visonlyqa_synthetic_splits, train_data_splits, train_data_text_splits
from src.path import get_evaluation_model_responses_path, get_evaluation_metrics_path
from src.prompts.prompts import get_image_token, get_postprocess_prompt
from src.evaluation.call_llm import call_llm
from src.evaluation.metrics import get_metrics
from src.utils import load_open_model, get_hf_dataset_name
from src.typing import QTYPE


class EvaluationTap(Tap):
    data: Literal["train", "eval_real", "eval_synthetic", "eval_synthetic_with_text"] = "eval_real"
    model: str
    prompt: str = "no_reasoning"
    seed: int = 68


# this does not work well
# def check_multiple_options_in_response(response: str, options: list[str]) -> bool:
#     # Create a pattern that matches any of the specified formats
#     pattern = r"[\s,]*".join([re.escape(option) for option in options])  # Handles "a,b,c,d" or "a, b, c, d"
#     pattern_parenthesis = r"[\s,]*".join([r"\(" + re.escape(option) + r"\)" for option in options])  # Handles "(a), (b), (c), (d)"

#     # Combine both patterns with OR to match either format
#     full_pattern = f"({pattern})|({pattern_parenthesis})"

#     # Use regex search to check if the pattern is present in the response
#     return bool(re.search(full_pattern, response))


def get_final_answer_from_response(prompt: str, response: str, question_type: QTYPE, response_options: list[str], seed: int) -> str:
    # easy cases
    if question_type == "single_answer":
        for parenthetis in [True, False]:
            found_option = []
            for option in response_options:
                if parenthetis:
                    option_str = f"({option})"
                else:
                    option_str = option
                
                if option_str in response:
                    found_option.append(option)
            if len(found_option) == 1:
                return found_option[0]
    # this does not work well
    # elif question_type == "multiple_answers":
    #     # automatic detection only for multiple answers (>=2) because detection for single answer is unreliable
    #     for option_num in range(len(response_options), 1, -1):
    #         for options in itertools.combinations(response_options, option_num):
    #             if check_multiple_options_in_response(response, options):
    #                 return ",".join(options)
    
    # extract using gpt-4o
    postprocess_prompt = get_postprocess_prompt(question=prompt, response=response, question_type=question_type, response_options=response_options)
    return call_llm(model_name="gpt-4o-2024-08-06", prompt=postprocess_prompt, image=None, seed=seed).replace(" ", "")


if __name__ == "__main__":
    args = EvaluationTap().parse_args()
    
    repository_name = get_hf_dataset_name(args.data)
    
    open_model, tokenizer, processor = load_open_model(args.model)
    
    splits_list = {
        "train": ["train_all_first_100"] + [f"{split}_50" for split in train_data_splits],
        "eval_real": visonlyqa_real_splits,
        "eval_synthetic": visonlyqa_synthetic_splits,
        "eval_synthetic_with_text": train_data_text_splits,
    }[args.data]
    
    if "finetuning_results" in args.model:
        splits_list = [split for split in splits_list if split in args.model]
    
    all_y_pred = []
    all_y_true = []
    for split in splits_list:
        print(f"Split: {split}")
        dataset = datasets.load_dataset(repository_name, split=split)
        
        # get responses
        responses_list = []
        y_pred = []  # extracted final answers
        for d in tqdm(dataset):
            prompt = get_image_token(args.model) + d[f"prompt_{args.prompt}"]
            image = d["decoded_image"]
            
            response = call_llm(model_name=args.model, prompt=prompt, image=image, image_path=d["image_path"],
                                open_model=open_model, hf_tokenizer=tokenizer, hf_processor=processor,
                                seed=args.seed)
            
            # postprocess (extract final answers)
            question_type = d["question_type"]
            response_options = d["response_options"]
            
            postprocessed_response = get_final_answer_from_response(
                prompt=prompt, response=response, question_type=question_type, response_options=response_options, seed=args.seed
            )
            if question_type == "single_answer":
                for option in response_options:
                    if option in postprocessed_response:
                        postprocessed_response = option
                        break

            responses_list.append({"prompt": prompt, "response": response, "y_pred": postprocessed_response, "y_true": d["answer"]})
            y_pred.append(postprocessed_response)            
            
        # save responses
        output_path = get_evaluation_model_responses_path(split=split, prompt=args.prompt, model_name=args.model, train_eval=args.data)
        with open(output_path, "w") as f:
            for response in responses_list:
                f.write(json.dumps(response) + "\n")
        
        # save postprocessed answers
        postprocessed_output_path = output_path.with_suffix(".postprocessed.json")
        with open(postprocessed_output_path, "w") as f:
            json.dump(y_pred, f, indent=2)
        
        # evaluation
        y_true = dataset["answer"]
        metrics = get_metrics(y_pred=y_pred, y_true=y_true)
        metrics_path = get_evaluation_metrics_path(split=split, prompt=args.prompt, model_name=args.model, train_eval=args.data)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        all_y_pred.extend(y_pred)
        all_y_true.extend(y_true)
    
    if "eval" in args.data:
        # evaluation for all splits
        metrics = get_metrics(y_pred=all_y_pred, y_true=all_y_true)
        metrics_path = get_evaluation_metrics_path(split="all", prompt=args.prompt, model_name=args.model, train_eval=args.data)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # make sure to release the memory
    open_model = None
    tokenizer = None
    torch.cuda.empty_cache()
