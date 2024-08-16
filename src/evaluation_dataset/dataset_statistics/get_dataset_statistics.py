import json

import datasets

from src.path import tables_dir, eval_real_dataset_stats_path, eval_synthetic_dataset_stats_path, eval_synthetic_with_text_dataset_stats_path
from src.utils import get_hf_dataset_name
from src.config import (
    visonlyqa_real_splits, visonlyqa_synthetic_splits, visonlyqa_synthetic_with_text_splits,
    eval_real_splits_capitalized, eval_synthetic_splits_capitalized, eval_synthetic_with_text_splits_capitalized
)


get_stat_label = {
    "num_examples": "\\# Examples",
}


if __name__ == "__main__":
    splits_dir = {"real": visonlyqa_real_splits, "synthetic": visonlyqa_synthetic_splits, "synthetic_with_text": visonlyqa_synthetic_with_text_splits}
    capitalized_dir = {"real": eval_real_splits_capitalized, "synthetic": eval_synthetic_splits_capitalized, "synthetic_with_text": eval_synthetic_with_text_splits_capitalized}
    
    for real_synthetic in splits_dir.keys():
        print(f"Processing {real_synthetic} dataset")
        
        repository_name = get_hf_dataset_name(f"eval_{real_synthetic}")
        
        dataset_stats: dict = {}
        for split in splits_dir[real_synthetic] + ["all"]:
            dataset = datasets.load_dataset(repository_name, split=split)
            
            dataset_stats[split] = {
                "num_examples": len(dataset),
            }
        
        # save dataset stats
        eval_dataset_stats_path = {"real": eval_real_dataset_stats_path, "synthetic": eval_synthetic_dataset_stats_path, "synthetic_with_text": eval_synthetic_with_text_dataset_stats_path}[real_synthetic]
        eval_dataset_stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_dataset_stats_path, "w") as f:
            json.dump(dataset_stats, f, indent=4)

        # create stats table (labex)
        table = []
        first_row = [""] + capitalized_dir[real_synthetic]
        for stat_name in ["num_examples"]:
            row = [get_stat_label[stat_name]]
            for split in splits_dir[real_synthetic] + ["all"]:
                value = dataset_stats[split][stat_name]
                if stat_name == "num_examples":
                    row.append(f"{value:d}")
                else:
                    raise NotImplementedError(f"Stat name {stat_name} is not supported")
            
            table.append(row)
        
        tables_dir.mkdir(parents=True, exist_ok=True)
        with open(tables_dir / f"visonlyqa-eval-{real_synthetic}_stats.tex", "w") as f:
            for row in [first_row] + table:
                f.write(" & ".join(row) + " \\\\\n")
