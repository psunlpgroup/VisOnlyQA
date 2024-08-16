# python src/analysis/download_open_vlm_leaderboard.py  # data is already included in the repository
python src/analysis/correlation_with_other_datasets.py

# annotation spreadsheet for error analysis on cot
python src/analysis/cot_error_analysis/generate_cot_annotation_spreadsheet.py
python src/analysis/cot_error_analysis/postproecess_cot_annotation.py
