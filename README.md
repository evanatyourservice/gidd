# Generalized Interpolating Discrete Diffusion

Dimitri von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, Thomas Hofmann

---




## Getting Started

### Setup

1. Set up environment:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

2. (optional) Log into W&B (`wandb login`) for experiment tracking or disable it (`wandb disabled`) if you don't need/want it.


### Training

To reproduce the training runs from the paper, you can use the following commands.
In this example, we are training on a single node with 8 GPUs, feel free to adjust the `--nnodes` and `--nproc_per_node` arguments to match your setup.
The checkpoints will be saved under `./outputs/{YYYY-MM-DD}/{HH-MM-SS}/checkpoints/` by default.


```bash
# GIDD+ (p_u = 0.0)
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd logging.run_name="'small-gidd+-owt-pu=0.0'"

# GIDD+ (p_0 > 0.0)
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name gidd model.p_uniform=0.1 logging.run_name="'small-gidd+-owt-pu=0.1'"

# MDLM baseline
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name mdlm logging.run_name="'small-mdlm-owt'"

# AR baseline
torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py --config-name ar logging.run_name="'small-ar-owt'"
```


### Inference

There are also a couple of scripts to run inference and evaluate the trained models.

#### Generate samples
The following command will generate `num_samples=16` samples in `num_denoising_steps=128` iterations from the model checkpoint located at `path` and save them to `samples_dir=samples.pt`.
```bash
python gidd/eval/generate_samples.py path=./outputs/path/to/checkpoint/ samples_dir=samples.pt num_samples=16 num_denoising_steps=128 batch_size=16
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
