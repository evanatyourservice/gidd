import torch
import torch.nn as nn
import torch.distributed as dist

from gidd.diffusion_process import sample_t, NoiseSchedule
from gidd.loss import Loss


class DiffusionTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, loss_fn: Loss, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn
        self.dtype = dtype

        self.device = next(model.parameters()).device

        self.register_buffer("pad_id", torch.tensor(tokenizer.pad_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("mask_id", torch.tensor(tokenizer.mask_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("t0", torch.zeros(1, device=self.device))
        self.register_buffer("t1", torch.ones(1, device=self.device))

    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)

    def forward(self, batch):
        batch_size = batch["input_ids"].size(0)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            t = sample_t(self.config, batch_size, device=self.device)
            z_t = self.noise_schedule.sample_zt(batch["input_ids"], t)

            logits = self.model(z_t, t)
            loss, _, metrics = self.loss_fn.forward(
                logits=logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                z_t=z_t,
                t=t,
                reduction=self.config.loss.reduction,
            )
        return loss, metrics


class AutoregressiveTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, loss_fn, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.dtype = dtype
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.device = next(model.parameters()).device
    
    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)

    def forward(self, batch):
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            labels = batch["input_ids"][:, 1:]
            loss_mask = batch["attention_mask"][:, :-1]

            logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            logits = logits[:, :-1]
            loss = self.loss_fn(logits.transpose(1, 2), labels)
            total_loss = (loss * loss_mask).sum()
            total_tokens = loss_mask.sum().float()

            if self.world_size > 1:
                dist.all_reduce(total_tokens)
                total_tokens /= self.world_size

            loss = total_loss / total_tokens

        return loss, {
            "elbo": loss.detach(),
            "nll": loss.detach(),
            "ppl": loss.detach().exp(),
        }


def get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype=None):
    if config.model.type == "diffusion":
        return DiffusionTrainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    elif config.model.type == "autoregressive":
        return AutoregressiveTrainer(config, model, tokenizer, loss_fn, dtype)
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")
