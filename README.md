# Generalized Interpolating Discrete Diffusion

By Dimitri von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, Thomas Hofmann

[![arXiv](https://img.shields.io/badge/arXiv-2503.04482-d22c2c.svg)](https://arxiv.org/abs/2503.04482)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Xv4RyZhXHkIpIZeMYahl_4kMthLxKdg_?usp=sharing)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-GIDD-f59a0c)](https://huggingface.co/collections/dvruette/generalized-interpolating-discrete-diffusion-67c6fc45663eafb85c6487af)

---

![animation](animation.gif)

We present Generalized Interpolating Discrete Diffusion (GIDD), a novel framework for training discrete diffusion models.
GIDD can be seen as a generalization of the popular masked diffusion paradigm (MDM) to any diffusion process that can be written as a linear interpolation between a data distribution and some (time-variable) mixing distribution.
We demonstrate the flexibility of GIDD by training models on a hybrid diffusion process that combines masking and uniform noise.
The model therefore is trained to not only "fill in the blanks" (i.e. the masked tokens), but also to consider the correctness of already-filled-in tokens and, if necessary, replace incorrect tokens with more plausible ones.
We show that GIDD models trained on hybrid noise have better sample quality (generative PPL) than mask-only models, and that they are able to identify and correct their own mistakes in generated samples through a self-correction step.
This repository contains all training and evaluation code necessary for reproducing the results in the paper.



### Pretrained Models
Our trained checkpoints are available on HuggingFace under the following links. All of them have been trained on 131B tokens from the [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) dataset with the [GPT-2 tokenizer](https://huggingface.co/openai-community/gpt2).

| Model | Small (169.6M) | Base (424.5M) |
|-------|-------|------|
| GIDD+ ($p_u = 0.0$) | [dvruette/gidd-small-p_unif-0.0](https://huggingface.co/dvruette/gidd-small-p_unif-0.0) | [dvruette/gidd-base-p_unif-0.0](https://huggingface.co/dvruette/gidd-base-p_unif-0.0) |
| GIDD+ ($p_u = 0.1$) | [dvruette/gidd-small-p_unif-0.1](https://huggingface.co/dvruette/gidd-small-p_unif-0.1) | [dvruette/gidd-base-p_unif-0.1](https://huggingface.co/dvruette/gidd-base-p_unif-0.1) |
| GIDD+ ($p_u = 0.2$) | [dvruette/gidd-small-p_unif-0.2](https://huggingface.co/dvruette/gidd-small-p_unif-0.2) | [dvruette/gidd-base-p_unif-0.2](https://huggingface.co/dvruette/gidd-base-p_unif-0.2) |


## Quick Start

0. Clone the repository (recommended version: [v1.1.0](https://github.com/dvruette/gidd/tree/v1.1.0)

1. Set up the environment:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

2. For quickly downloading a trained model and playing around with it, the `GiddPipeline` class is most convenient:

```python
from gidd import GiddPipeline

# Download a pretrained model from HuggingFace
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
pipe.to(device)

# Generate samples
texts = pipe.generate(num_samples=4, num_inference_steps=128)

# Run self-correction step
corrected_texts = pipe.self_correction(texts, num_inference_steps=128, early_stopping=True, temperature=0.1)

print(corrected_texts)
```


## Reproducing Experiments

### Training


To reproduce the training runs from the paper, you can use the following commands.
In this example, we are training on a single node with 8 GPUs, feel free to adjust the `--nnodes` and `--nproc_per_node` arguments to match your setup.
The checkpoints will be saved under `./outputs/{YYYY-MM-DD}/{HH-MM-SS}/checkpoints/` by default.

(optional) Log into W&B with `wandb login` for experiment tracking or disable via `wandb disabled` if you don't need/want it.

```bash
# GIDD+ (p_u = 0.0)
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd logging.run_name="'small-gidd+-owt-pu=0.0'"

# GIDD+ (p_u = 0.0) with QUAD optimizer
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd optimizer=quad training.warmup_steps=100 logging.run_name="'small-gidd+-owt-pu=0.0-quad'"

# GIDD+ (p_0 > 0.0)
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd model.p_uniform=0.1 logging.run_name="'small-gidd+-owt-pu=0.1'"

# GIDD+ (p_0 > 0.0) with QUAD optimizer
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd model.p_uniform=0.1 optimizer=quad training.warmup_steps=0 logging.run_name="'small-gidd+-owt-pu=0.1-quad'"

# MDLM baseline
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name mdlm logging.run_name="'small-mdlm-owt'"

# AR baseline
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name ar logging.run_name="'small-ar-owt'"
```


### Evaluation

There are also a couple of scripts to run inference and evaluate the trained models.
Note that these scripts expect the checkpoint format that is saved by the training script, so the checkpoints from HuggingFace are not directly compatible.
You can download our original training checkpoints from here: https://polybox.ethz.ch/index.php/s/BbxZcYDSoXf8aL4

#### Generate samples
The following command will generate `num_samples=16` samples in `num_denoising_steps=128` iterations from the model checkpoint located at `path` and save them to `samples_dir=samples.pt`.
```bash
python gidd/eval/generate_samples.py path=./outputs/path/to/checkpoint/ samples_path=samples.pt num_samples=16 num_denoising_steps=128 batch_size=16
```

#### Generative PPL
Given a file containing samples generated with the `generate_samples.py` script, the following command will compute the generative PPL.
Here we assume that the diffusion model used to generate samples located at `samples.pt` uses the `gpt2` tokenizer, and we compute generative PPL using `google/gemma-2-9b` as a reference model (note that `gemma-2-9b` requires you to log into your HF account using `huggingface-cli login`).
The results will be saved to `metrics_path=metrics.json`.
```bash
python gidd/eval/generative_ppl.py samples_path=samples.pt model_tokenizer=gpt2 pretrained_model=google/gemma-2-9b batch_size=4 metrics_path=metrics.json
```

#### Validation loss
A simple helper script to compute the loss of a trained model on the entire validation split.
```bash
python gidd/eval/loss.py path=./outputs/path/to/checkpoint/ batch_size=32
```

#### Self-correction
This script will run the self-correction step on the samples contained in `samples.pt` (e.g. generated with the `generate_samples.py` script) and save the corrected samples to `corrected_samples.pt`.
The `temp` argument controls the temperature used when resampling tokens from the model (see paper for more details).
```bash
python gidd/eval/self_correction.py path=./outputs/path/to/checkpoint/ samples_path=samples.pt corrected_samples_path=corrected_samples.pt batch_size=16 num_denoising_steps=128 temp=0.1
```
