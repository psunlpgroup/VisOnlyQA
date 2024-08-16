""" Get dataset statistics for training dataset. """


import json

from src.path import train_dataset_stats_path, train_dataset_dir
from src.config import train_data_splits


get_stat_label = {
    "num_examples": "\\# Examples",
}


if __name__ == "__main__":
    dataset_stats: dict = {}
    train_data_info_path = train_dataset_dir / "internvl_meta/all.json"
    with open(train_data_info_path, "r") as f:
        train_data_info = json.load(f)
    
    for split in train_data_splits:
        dataset_stats[split] = {
            "num_examples": train_data_info[split]["length"]
        }
    
    # save dataset stats
    train_dataset_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_dataset_stats_path, "w") as f:
        json.dump(dataset_stats, f, indent=4)
