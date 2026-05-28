---
library_name: peft
license: other
base_model: /root/autodl-tmp/models/Qwen/Qwen3-8B
tags:
- base_model:adapter:/root/autodl-tmp/models/Qwen/Qwen3-8B
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3-8b-atc-lora-fast
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-8b-atc-lora-fast

This model is a fine-tuned version of [/root/autodl-tmp/models/Qwen/Qwen3-8B](https://huggingface.co//root/autodl-tmp/models/Qwen/Qwen3-8B) on the atc_task1_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4034

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 8
- eval_batch_size: 4
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.05
- num_epochs: 2.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.3153        | 2.0   | 256  | 0.4034          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.5.1+cu124
- Datasets 4.0.0
- Tokenizers 0.22.2