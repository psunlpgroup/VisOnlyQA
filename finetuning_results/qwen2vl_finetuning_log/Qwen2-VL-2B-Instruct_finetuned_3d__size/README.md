---
library_name: transformers
license: other
base_model: Qwen/Qwen2-VL-2B-Instruct
tags:
- llama-factory
- full
- generated_from_trainer
model-index:
- name: Qwen2-VL-2B-Instruct_finetuned_3d__size
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Qwen2-VL-2B-Instruct_finetuned_3d__size

This model is a fine-tuned version of [Qwen/Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) on the visonlyqa_3d__size dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0000

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- total_eval_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 3.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.6572        | 0.1286 | 10   | 0.0645          |
| 0.5123        | 0.2572 | 20   | 0.0525          |
| 0.2697        | 0.3859 | 30   | 0.0065          |
| 0.0544        | 0.5145 | 40   | 0.0022          |
| 0.0224        | 0.6431 | 50   | 0.0046          |
| 0.0311        | 0.7717 | 60   | 0.0076          |
| 0.0347        | 0.9003 | 70   | 0.0013          |
| 0.0164        | 1.0289 | 80   | 0.0025          |
| 0.0259        | 1.1576 | 90   | 0.0015          |
| 0.0255        | 1.2862 | 100  | 0.0079          |
| 0.0895        | 1.4148 | 110  | 0.0022          |
| 0.0285        | 1.5434 | 120  | 0.0026          |
| 0.0101        | 1.6720 | 130  | 0.0080          |
| 0.0296        | 1.8006 | 140  | 0.0002          |
| 0.0158        | 1.9293 | 150  | 0.0005          |
| 0.022         | 2.0579 | 160  | 0.0001          |
| 0.0105        | 2.1865 | 170  | 0.0001          |
| 0.0048        | 2.3151 | 180  | 0.0000          |
| 0.0044        | 2.4437 | 190  | 0.0000          |
| 0.0023        | 2.5723 | 200  | 0.0000          |
| 0.003         | 2.7010 | 210  | 0.0000          |
| 0.0025        | 2.8296 | 220  | 0.0000          |
| 0.0001        | 2.9582 | 230  | 0.0000          |


### Framework versions

- Transformers 4.46.1
- Pytorch 2.5.0+cu124
- Datasets 2.21.0
- Tokenizers 0.20.3
