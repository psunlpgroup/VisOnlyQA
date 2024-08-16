from src.typing import VisonlyQA_Instance, TRAIN_VAL_TEST
from src.prompts import get_evaluation_prompt


def get_data_instance(split: TRAIN_VAL_TEST, image_path: str, answer: str, question: str, options: list[str], image_category: str, task_category: str, dataset_name: str, id_str: str) -> VisonlyQA_Instance:
    return {
        "image_path": image_path,
        "question": question if split == "test" else None,
        "answer": answer,
        "prompt_reasoning": get_evaluation_prompt("reasoning", question, response_options=options) if split == "test" else None,
        "prompt_no_reasoning": get_evaluation_prompt("no_reasoning", question, response_options=options),
        "image_category": image_category,
        "task_category": task_category,
        "question_type": "single_answer",
        "response_options": options,
        "source": dataset_name,
        "id": id_str,
    }
