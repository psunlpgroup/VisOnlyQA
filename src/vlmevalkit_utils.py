from src.path import vlmevalkit_models_list_path
from src.config import open_models_with_specific_code_list


def get_vlmeval_models_list() -> list[str]:
    """ Get the list of models that are in the VLMEvalKit models list """
    
    output_list: list[str] = []
    with open(vlmevalkit_models_list_path, "r") as f:
        for line in f:
            output_list.append(line.strip())
    
    return output_list


def is_vlmeval_models(model_name: str) -> bool:
    """ Check if the model can be evaluated using VLMEvalKit """
    
    # we use model specific code for these models
    for ignore_name in open_models_with_specific_code_list:
        if ignore_name in model_name:
            return False
    
    # for other models, check if they are in the VLMEvalKit models list
    return model_name in get_vlmeval_models_list()
