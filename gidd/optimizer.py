import torch
import heavyball
from gidd.quad import QUAD


def get_optimizer(config, trainer):
    params = trainer.parameters()
    if config.optimizer.type == "adam":
        optimizer = torch.optim.AdamW(params, betas=tuple(config.optimizer.betas), weight_decay=config.optimizer.weight_decay, eps=config.optimizer.eps)
    elif config.optimizer.type == "psgd":
        # heavyball.utils.compile_mode = None
        heavyball.utils.set_torch()
        optimizer = heavyball.ForeachPSGDKron(params, beta=config.optimizer.beta, weight_decay=config.optimizer.weight_decay, mars=config.optimizer.mars, caution=config.optimizer.caution)
        optimizer.promote = True
        # heavyball.utils.fused_hook(params, heavyball.ForeachPSGDKron, beta=config.optimizer.beta, mars=config.optimizer.mars)
    elif config.optimizer.type == "quad":
        dtype_str = config.optimizer.dtype
        dtype = None
        if dtype_str is not None:
            if dtype_str == "float32":
                dtype = torch.float32
            elif dtype_str == "float64":
                dtype = torch.float64
            elif dtype_str == "bfloat16":
                dtype = torch.bfloat16
            elif dtype_str == "float16":
                dtype = torch.float16
        optimizer = QUAD(
            params,
            lr=config.optimizer.lr,
            lr_style=config.optimizer.lr_style,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            preconditioner_lr=config.optimizer.preconditioner_lr,
            max_size_dense=config.optimizer.max_size_dense,
            max_skew_dense=config.optimizer.max_skew_dense,
            normalize_grads=config.optimizer.normalize_grads,
            dtype=dtype,
        )
    return optimizer
