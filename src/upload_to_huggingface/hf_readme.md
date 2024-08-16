# VisOnlyQA

This repository contains the code and data for the paper "VisOnlyQA: Large Vision Language Models Still Struggle with Visual Perception of Geometric Information".

VisOnlyQA is designed to evaluate the visual perception capability of large vision language models (LVLMs) on geometric information of scientific figures. The evaluation set includes 1,200 mlutiple choice questions in 12 visual perception tasks on 4 categories of scientific figures. We also provide a training dataset consisting of 70k instances.

* Datasets:
  * Eval-Real: [https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real)
  * Eval-Synthetic: [https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Synthetic](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Synthetic)
  * Train: [https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Train](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Train)
* Code: [https://github.com/psunlpgroup/VisOnlyQA](https://github.com/psunlpgroup/VisOnlyQA)

<p align="center">
<img src="readme_figures/accuracy_radar_chart.png" width="500">
</p>

```bibtex
@misc{kamoi2024visonlyqa,
    title={VisOnlyQA: Large Vision Language Models Still Struggle with Visual Perception of Geometric Information}, 
    author={Ryo Kamoi and Yusen Zhang and Sarkar Snigdha Sarathi Das and Ranran Haoran Zhang and Rui Zhang},
    year={2024},
}
```

## Dataset

The dataset is provided in Hugging Face Dataset.

* Eval-Real: [https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Real)
  * 500 instances for questions on figures in existing datasets (e.g., MathVista, MMMU, and CharXiv)
* Eval-Synthetic: [https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Synthetic](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Eval_Synthetic)
  * 700 instances for questions on synthetic figures
* Train: [https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Train](https://huggingface.co/datasets/ryokamoi/VisOnlyQA_Train)
  * 70,000 instances for training (synthetic figures)

[dataset](https://github.com/psunlpgroup/VisOnlyQA/tree/main/dataset) folder of the GitHub repository includes identical datasets, except for the training data.

### Examples

<p align="center">
<img src="readme_figures/examples.png" width="800">
</p>

### Usage

```python
from datasets import load_dataset

real_eval = load_dataset("ryokamoi/VisOnlyQA_Eval_Real")
real_synthetic = load_dataset("ryokamoi/VisOnlyQA_Eval_Synthetic")

# Splits
print(real_eval.keys())
# dict_keys(['geometry__triangle', 'geometry__quadrilateral', 'geometry__length', 'geometry__angle', 'geometry__area', 'geometry__diameter_radius', 'chemistry__shape_single', 'chemistry__shape_multi', 'charts__extraction', 'charts__intersection'])

print(real_synthetic.keys())
# dict_keys(['syntheticgeometry__triangle', 'syntheticgeometry__quadrilateral', 'syntheticgeometry__length', 'syntheticgeometry__angle', 'syntheticgeometry__area', '3d__size', '3d__angle'])

# Prompt
print(real_eval['geometry__triangle'][0]['prompt_no_reasoning'])
# There is no triangle ADP in the figure. True or False?

# A triangle is a polygon with three edges and three vertices, which are explicitly connected in the figure.

# Your response should only include the final answer (True, False). Do not include any reasoning or explanation in your response.

# Image
print(real_eval['geometry__triangle'][0]['decoded_image'])
# <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=103x165 at 0x7FB4F83236A0>

# Answer
print(real_eval['geometry__triangle'][0]['answer'])
# False
```

### Data Format

Each instance of VisOnlyQA dataset has the following attributes:

#### Features
* `decoded_image`: [PIL.Image] Input image
* `question`: [string] Question (without instruction)
* `prompt_reasoning`: [string] Prompt with intstruction to use chain-of-thought
* `prompt_no_reasoning`: [string] Prompt with intstruction **not** to use chain-of-thought
* `answer`: [string] Correct answer (e.g., `True`, `a`)

#### Metadata
* `image_path`: [string] Path to the image file
* `image_category`: [string] Category of the image (e.g., `geometry`, `chemistry`)
* `question_type`: [string] `single_answer` or `multiple answers`
* `task_category`: [string] Category of the task (e.g., `triangle`)
* `response_options`: [List[string]] Multiple choice options (e.g., `['True', 'False']`, `['a', 'b', 'c', 'd', 'e']`)
* `source`: [string] Source dataset
* `id`: [string] Unique ID

### Statistics

<p align="center">
<img src="readme_figures/stats.png" width="800">
</p>

## License

Please refer to [LICENSE.md](./LICENSE.md).

## Contact

If you have any questions, feel free to open an issue or reach out directly to [Ryo Kamoi](https://ryokamoi.github.io/) (ryokamoi@psu.edu).
