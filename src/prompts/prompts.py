import random
from typing import Literal

from src.config import model_image_tokens_dict
from src.typing import QTYPE
from src.vlmevalkit_utils import is_vlmeval_models


template = """{question}

"""


NO_REASONING_PROMPT = template + "Your response should only include the final answer ({response_type}). Do not include any reasoning or explanation in your response."


REASONING_PROMPT = template + "In your response, provide a short explanation or reasoning for your answer. Then, provide the final answer ({response_type})."


prompt_dict = {
    "no_reasoning": NO_REASONING_PROMPT,
    "reasoning": REASONING_PROMPT
}


def get_image_token(model_name: str):
    if "gpt" in model_name or "claude" in model_name or "gemini" in model_name:
        return ""
    
    if "InternVL" in model_name:
        image_token = model_image_tokens_dict["InternVL"]
        return f"{image_token}\n"
    
    if "Phi-3.5" in model_name:
        image_token = model_image_tokens_dict["Phi-3.5"]
        return f"{image_token}\n"
    
    if "Qwen2-VL" in model_name:
        return ""
    
    if is_vlmeval_models(model_name):
        return ""
    
    raise NotImplementedError(f"Model {model_name} is not supported")
    

def get_evaluation_prompt(prompt_name: str, question: str, response_options: list[str]) -> str:
    prompt = prompt_dict[prompt_name]
    return prompt.format(question=question, response_type=", ".join(response_options))


def shuffle_options(question: str, answer: str, options: list[str], question_type: Literal["single_answer", "multiple_answers"]) -> tuple[str, str]:
    if question_type == "multiple_answers":
        # we do not shuffle the order of options
        return question, answer
    
    if len(options) == 2:
        # we do not change the order of True/False options
        return question, answer
    
    shuffled_options = random.Random(question).sample(options, len(options))
    
    # extract options from the question
    options_dict = {}
    for option in options[::-1]:
        option_index = question.rfind(f"({option})")
        option_str = question[option_index+3:]
        if option == options[-1]:
            option_str += " "
        options_dict[option] = option_str
        
        question = question[:option_index]
    
    # shuffle options
    for idx, original_option in enumerate(shuffled_options):
        option = options[idx]
        question += f"({option}){options_dict[original_option]}"
    
    # convert answer
    answer = options[shuffled_options.index(answer)]
    
    return question, answer


POSTPROCESS_RESPONSE = """Your task is to extract the final answer (selected option) from the response. Your response should only include {response_type}.

Question: {question}

Response: {response}"""


POSTPROCESS_RESPONSE_MULTIPLE_ANSWERS = """Your task is to extract the final answer from the response. Your response should only include the final answer(s) in a format of "a", "a,b", "a,c,d", "a,b,c,d".
For example, "(a), (b), (c), (d)" should be converted to "a,b,c,d".

Response: {response}"""


def get_postprocess_prompt(question: str, response: str, question_type: QTYPE, response_options: list[str]):
    if question_type == "multiple_answers":
        return POSTPROCESS_RESPONSE_MULTIPLE_ANSWERS.format(response=response)
    else:
        assert question_type == "single_answer"
        
        response_type_str = ", ".join(response_options)
        return POSTPROCESS_RESPONSE.format(response_type=response_type_str, question=question, response=response)
