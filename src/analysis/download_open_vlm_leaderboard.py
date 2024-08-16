# raise Exception("The repository already contains the OpenVLM Leaderboard we used for the analysis, which was downloaded on Sep 16, 2024.")
# https://huggingface.co/spaces/opencompass/open_vlm_leaderboard

import json

from gradio_client import Client

from src.path import openvlm_leaderboard_dir


def get_open_vlm_leaderboard(datasets_list: str):
    client = Client("opencompass/open_vlm_leaderboard")
    result = client.predict(
            fields=datasets_list,
            model_size=["<4B","4B-10B","10B-20B","20B-40B",">40B","Unknown"],
            model_type=["API","OpenSource","Proprietary"],
            api_name="/filter_df"
    )
    
    return result



if __name__ == "__main__":
    openvlm_leaderboard_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_list = ["MMBench_V11","MMStar","MMMU_VAL","MathVista","OCRBench","AI2D","HallusionBench","MMVet", "SEEDBench_IMG","MME","LLaVABench","CCBench","RealWorldQA","POPE","ScienceQA_TEST","SEEDBench2_Plus","MMM-Bench_VAL","BLINK"]
    result = get_open_vlm_leaderboard(datasets_list)
    
    with open(openvlm_leaderboard_dir / "openvlm.json", "w") as f:
        json.dump(result, f, indent=4)
