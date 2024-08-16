from typing import Literal, TypedDict


TRAIN_VAL_TEST = Literal["train", "val", "test"]

QTYPE = Literal["single_answer", "multiple_answers"]


class VisonlyQA_Instance(TypedDict):
    image_path: str
    question: str
    answer: str
    prompt_reasoning: str
    prompt_no_reasoning: str
    image_category: str
    task_category: str
    question_type: QTYPE
    response_options: list[str]
    source: str
    id: str
