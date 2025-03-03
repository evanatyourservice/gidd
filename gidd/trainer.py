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
            z_t, target_features = self.noise_schedule.sample_zt(batch["input_ids"], t)

            pred_features = self.model(z_t, t)
            loss, _, metrics = self.loss_fn.forward(
                pred_features=pred_features,
                target_features=target_features,
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
            loss = self.loss_fn(logits.flatten(0, 1), labels.flatten(0, 1)).view_as(labels)
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
    
    @torch.no_grad()
    def compute_nll(self, batch, reduce_metrics=False, return_token_nlls=False):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        labels = batch["input_ids"][:, 1:]
        loss_mask = batch["attention_mask"][:, :-1]

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits[:, :-1]
            loss = self.loss_fn(logits.flatten(0, 1), labels.flatten(0, 1)).view_as(labels)
        
        total_nll = (loss * loss_mask).sum()
        total_tokens = loss_mask.sum()
        total_batch_size = torch.tensor(batch["input_ids"].size(0), device=self.device)

        if reduce_metrics and self.world_size > 1:
            dist.all_reduce(total_nll)
            dist.all_reduce(total_tokens)
            dist.all_reduce(total_batch_size)

        nll = total_nll / total_tokens
        seq_nll = total_nll / total_batch_size

        metrics = {
            "elbo": nll,
            "nll": nll,
            "ppl": nll.exp(),
            "seq_nll": seq_nll,
            "seq_ppl": seq_nll.exp(),
        }

        return (metrics, loss) if return_token_nlls else metrics


def get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype=None):
    if config.model.type == "diffusion":
        return DiffusionTrainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    elif config.model.type == "autoregressive":
        return AutoregressiveTrainer(config, model, tokenizer, loss_fn, dtype)
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")
