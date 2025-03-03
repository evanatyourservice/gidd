from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gidd.utils import sample_categorical


def sample_t(config, batch_size, eps=None, device=None):
    if eps is None:
        eps = config.model.t_eps

    if config.training.low_discrepancy_sampling:
        t = torch.arange(batch_size, device=device) / batch_size
        t = (t + torch.rand(1, device=device)).fmod(1.0)
    else:
        t = torch.rand(batch_size, device=device)

    t = (1 - 2 * eps) * t + eps
    return t


class NoiseSchedule(nn.Module, ABC):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)

        self.register_buffer("log_prior", self.get_log_prior())

    def get_log_prior(self):
        pr = torch.full((self.vocab_size,), -1e3)
        pr[self.mask_id] = 0
        return pr - pr.logsumexp(-1, keepdim=True)
    
    def sample_prior(self, shape):
        return torch.full(shape, self.mask_id, dtype=torch.long, device=self.log_prior.device)
    
    def get_features(self, input_ids, dtype=None):
        return F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=dtype)
    
    @abstractmethod
    def logits_at_t(self, features, t):
        raise NotImplementedError
    
    @abstractmethod
    def probs_at_t(self, prs, t):
        raise NotImplementedError

    def sample_zt(self, input_ids, t):
        features = self.get_features(input_ids, dtype=t.dtype)
        logits = self.logits_at_t(features, t)
        z_t = sample_categorical(logits.softmax(-1))
        return z_t, features


class HybridDiffusion(NoiseSchedule):
    def __init__(self, tokenizer, clip_noise=20, gamma=1.0, p_uniform=0.0):
        super().__init__(tokenizer)
        self.clip_noise = clip_noise
        self.p_uniform = max(np.exp(-clip_noise), p_uniform)

        log_B = -np.log1p((1 - self.p_uniform) / self.p_uniform * self.vocab_size / 2)
        mask = torch.zeros(self.vocab_size)
        mask[self.mask_id] = 1
        self.register_buffer("mask", mask, persistent=False)
        self.register_parameter("log_B", nn.Parameter(torch.tensor(float(log_B)).clip(-clip_noise), requires_grad=True))
        self.register_parameter("log_gamma", nn.Parameter(torch.tensor(float(gamma)).log(), requires_grad=True))
    
    def get_alpha_betapi(self, t, eps=1e-4):
        t = t[:, None]
        t1m = 1 - t

        gamma = self.log_gamma.exp()
        # .pow() autocasts to fp32
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)

        B = self.log_B.exp()
        c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
        C_t = t_gamma + t1m_gamma + (self.vocab_size - 2) * c_t
        # C_t should never be much smaller than 1,
        # but just in case it is, we clip it to avoid numerical instability
        C_t = C_t.clip(eps)

        alpha_t = (t1m_gamma - c_t) / C_t
        beta_pi = (t_gamma * self.mask + c_t * (1 - self.mask)) / C_t
        return alpha_t, beta_pi

    def logits_at_t(self, features, t):
        t = t[..., None, None]
        gamma = self.log_gamma.exp().to(t.dtype)
        log_B = self.log_B.to(t.dtype)
        xi_t = gamma / 2 * torch.log((1 - t) / t).clip(-self.clip_noise, self.clip_noise)
        logits = features.mul(xi_t - log_B)
        logits.add_(log_B)
        logits[..., self.mask_id] = -xi_t.squeeze(-1).expand_as(logits[..., self.mask_id])
        return logits
    
    def probs_at_t(self, prs, t, eps=1e-4):
        orig_dtype = prs.dtype
        t = t[:, None]
        t1m = 1 - t

        gamma = self.log_gamma.exp()
        # .pow() autocasts to fp32
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)

        B = self.log_B.exp()
        c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
        C_t = t_gamma + t1m_gamma + (self.vocab_size - 2) * c_t
        # C_t should never be much smaller than 1, but just in case it is, we clip it to avoid numerical instability
        C_t = C_t.clip(eps)

        alpha_t = (t1m_gamma - c_t) / C_t

        # beta_pi_hat = (t_gamma * mask + c_t * (1 - mask)) / C_t
        probs = prs.mul(alpha_t.unsqueeze(-1))
        probs.add_((c_t / C_t).unsqueeze(-1))
        probs[..., self.mask_id] = t_gamma / C_t
        return probs.to(orig_dtype)
    

class MaskedDiffusion(NoiseSchedule):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # required to be able to interchangeably mix our/mdlm schedule/loss
        self.register_buffer("log_gamma", torch.tensor(0.0))
        self.register_buffer("log_B", torch.tensor(-20.0))

    def get_sigmas(self, t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma

    def logits_at_t(self, features, t):
        _, sigma = self.get_sigmas(t)
        move_chance = 1 - torch.exp(-sigma)
        log_1m_move_chance = -sigma
        logits = (features + 1e-8).clip(1e-8).log().log_softmax(-1) + log_1m_move_chance[..., None, None]
        logits[:, :, self.mask_id] = move_chance.log().clip(-1e6)[..., None]
        return logits
    
    def probs_at_t(self, prs, t):
        _, sigma = self.get_sigmas(t)
        alpha_t = torch.exp(-sigma)
        probs = alpha_t[..., None, None] * prs
        probs[..., self.mask_id] = 1 - alpha_t.unsqueeze(-1)
        return probs

    def sample_zt(self, input_ids, t):
        _, sigma = self.get_sigmas(t)
        move_chance = 1 - torch.exp(-sigma)
        is_masked = torch.rand_like(input_ids.float()) < move_chance.unsqueeze(-1)
        z_t = torch.where(is_masked, self.mask_id, input_ids)
        return z_t, self.get_features(input_ids)


def get_noise_schedule(config, tokenizer):
    if config.model.diffusion_process == "gidd":
        noise_schedule = HybridDiffusion(tokenizer, p_uniform=config.model.p_uniform)
    elif config.model.diffusion_process == "mdlm":
        noise_schedule = MaskedDiffusion(tokenizer)
    else:
        raise ValueError(f"Unknown diffusion process: {config.model.diffusion_process}")

    return noise_schedule
