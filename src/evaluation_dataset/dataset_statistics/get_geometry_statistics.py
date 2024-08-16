import csv
import json

import numpy as np
import datasets

from src.path import test_intermediate_dir, dataset_stats_dir, tables_dir, alphageometry_intermediate_dir
from src.config import visonlyqa_synthetic_splits
from src.utils import get_hf_dataset_name


stats_names_list = ["points", "lines", "circles"]

if __name__ == "__main__":
    # postprocess annotation for real
    print("Postprocessing annotation for real")
    
    annotation_path = test_intermediate_dir / "statistics_annotation" / "VisOnlyQA_geometry_diagram_annotation - MathVista.csv"
    with open(annotation_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    stats_indices = {}
    for stat_name in stats_names_list:
        stats_indices[stat_name] = header.index(stat_name)
    
    real_stats_list_dict = {stat_name: [] for stat_name in stats_names_list}
    for row in rows:
        for stat_name in stats_names_list:
            count_str = row[stats_indices[stat_name]]
            if count_str == "":
                break
            
            real_stats_list_dict[stat_name].append(int(count_str))
    
    # synthetic geometric shapes
    print("Postprocessing synthetic geometric shapes")
    
    synthetic_stats_list_dict = {stat_name: [] for stat_name in stats_names_list}
    
    synthetic_geometry_splits = [s for s in visonlyqa_synthetic_splits if "geometry" in s]
    repository_name = get_hf_dataset_name("eval_synthetic")
    for split in synthetic_geometry_splits:
        dataset = datasets.load_dataset(repository_name, split=split)
        for d in dataset:
            data_id_full: str = d["id"]
            data_id = "_".join(data_id_full.split("_")[:2])  # format: syntheticgeometry_00000
            info_path = alphageometry_intermediate_dir / "info" / f"{data_id}.json"
            
            with open(info_path, "r") as f:
                info = json.load(f)
                synthetic_stats_list_dict["points"].append(len(info["point_positions"]))
                synthetic_stats_list_dict["lines"].append(len(info["line_instances"]))
                synthetic_stats_list_dict["circles"].append(len(info["circle_instances"]))
    
    # calculate stats
    stats_all: dict[str, dict[str, float]] = {}
    for real_synthetic, stats_list_dict in {"real": real_stats_list_dict, "synthetic": synthetic_stats_list_dict}.items():
        stats_all[real_synthetic] = {
            "average": {stat_name: np.mean(stats_list).item() for stat_name, stats_list in stats_list_dict.items()},
            "std": {stat_name: np.std(stats_list).item() for stat_name, stats_list in stats_list_dict.items()},
            "num": len(stats_list_dict[stats_names_list[0]]),
        }
    
    # save dataset stats
    save_dir = dataset_stats_dir / "geometric_shapes"
    save_dir.mkdir(parents=True, exist_ok=True)
    for real_synthetic, stats in stats_all.items():
        with open(save_dir / f"eval_{real_synthetic}_stats.json", "w") as f:
            json.dump(stats, f, indent=4)
    
    # create latex table
    table: list[list[str]] = [[""] + [f"\\# {stat_name.capitalize()}" for stat_name in stats_names_list]]
    for real_synthetic, stats in stats_all.items():
        table.append([real_synthetic.capitalize()] + [f"{stats['average'][stat_name]:.1f} ($\\pm$ {stats['std'][stat_name]:.1f})" for stat_name in stats_names_list])
    
    geometry_stats_table_dir = tables_dir / "geometry_stats"
    geometry_stats_table_dir.mkdir(parents=True, exist_ok=True)
    with open(geometry_stats_table_dir / "geometry_stats_table.tex", "w") as f:
        for row in table:
            f.write(" & ".join(row) + " \\\\\n")
