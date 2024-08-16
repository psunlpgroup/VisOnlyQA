""" Save the list of models supported by VLMEvalKit to a file. """


from VLMEvalKit.vlmeval.config import supported_VLM
from src.path import vlmevalkit_models_list_path


if __name__ == "__main__":
    vlmevalkit_models_list_path.parent.mkdir(parents=True, exist_ok=True)
    
    keys = list(supported_VLM.keys())
    with open(vlmevalkit_models_list_path, "w") as f:
        for key in keys:
            f.write(f"{key}\n")
