import math
import torch


class QUAD(torch.optim.Optimizer):
    """PSGD-QUAD optimizer.

    Args:
        params: list of parameters to optimize
        lr: learning rate
        lr_style: "adam" (default), "mu-p", or None, "adam" scales update norm to match adam's,
            "mu-p" scales update norm according to grad.shape[-2], None uses raw PSGD scaling (RMS=1.0).
        momentum: momentum beta
        weight_decay: weight decay
        preconditioner_lr: preconditioner learning rate
        max_size_dense: dimensions larger than this will have diagonal preconditioners, otherwise
            dense.
        max_skew_dense: dimensions with skew larger than this compared to the other dimension will
            have diagonal preconditioners, otherwise dense.
        normalize_grads: normalize incoming gradients to unit norm.
        dtype: dtype for all computations and states in QUAD.
    """
    def __init__(
        self,
        params: list[torch.nn.Parameter],
        lr: float = 0.001,
        lr_style: str | None = "adam",
        momentum: float = 0.95,
        weight_decay: float = 0.001,
        preconditioner_lr: float = 0.75,
        max_size_dense: int = 8192,
        max_skew_dense: float = 1.0,
        normalize_grads: bool = False,
        dtype: torch.dtype | None = None,
    ):
        defaults = dict(
            lr=lr,
            lr_style=lr_style,
            momentum=momentum,
            weight_decay=weight_decay,
            preconditioner_lr=preconditioner_lr,
            max_size_dense=max_size_dense,
            max_skew_dense=max_skew_dense,
            normalize_grads=normalize_grads,
            dtype=dtype,
        )
        super().__init__(params, defaults)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        momentum_buffers,
        merged_shapes,
        Qs,
        Ls,
        diags,
        mu_ps,
        state_steps,
    ):
        group_dtype = group['dtype']
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad if group_dtype is None else p.grad.to(dtype=group_dtype))
    
        if group["normalize_grads"]:
            torch._foreach_div_(grads, torch._foreach_add_(torch._foreach_norm(grads), 1e-7))
    
        for p, g in zip(params_with_grad, grads):
            state = self.state[p]
            dtype = g.dtype

            mu_ps.append(g.shape[-2] if len(g.shape) > 1 else None)
    
            if "momentum_buffer" not in state:
                state["step"] = torch.tensor(0, dtype=torch.int32, device=g.device)
                state["momentum_buffer"] = g.clone()
                state["merged_shape"] = merge_dims(state["momentum_buffer"])
                g_reshaped = state["momentum_buffer"].view(state["merged_shape"])
                scale = ((torch.mean((torch.abs(g_reshaped))**2))**(-1/4))**(1/2 if len(g_reshaped.shape) > 1 else 1.0)
                if g_reshaped.ndim <= 1:
                    state["Q"] = [scale * torch.ones_like(g_reshaped, dtype=dtype)]
                    state["L"] = [torch.zeros([], dtype=dtype, device=g_reshaped.device)]
                    state["diag"] = [True]
                else:
                    Qs_new = []
                    Ls_new = []
                    diag_new = []
                    for size in g_reshaped.shape:
                        if size > group["max_size_dense"] or size**2 > group["max_skew_dense"] * g_reshaped.numel():
                            Qs_new.append(scale * torch.ones(size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros([], dtype=dtype, device=g_reshaped.device))
                            diag_new.append(True)
                        else:
                            Qs_new.append(scale * torch.eye(size, dtype=dtype, device=g_reshaped.device))
                            Ls_new.append(torch.zeros([], dtype=dtype, device=g_reshaped.device))
                            diag_new.append(False)
                    state["Q"] = Qs_new
                    state["L"] = Ls_new
                    state["diag"] = diag_new
    
            momentum_buffers.append(state['momentum_buffer'])
            merged_shapes.append(state["merged_shape"])
            Qs.append(state["Q"])
            Ls.append(state["L"])
            diags.append(state["diag"])
            state_steps.append(state["step"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            momentum_buffers: list[torch.Tensor] = []
            merged_shapes: list[tuple] = []
            Qs: list[list | None] = []
            Ls: list[list | None] = []
            diags: list[list | None] = []
            mu_ps: list[float] = []
            state_steps: list[int] = []

            self._init_group(
                group,
                params_with_grad,
                grads,
                momentum_buffers,
                merged_shapes,
                Qs,
                Ls,
                diags,
                mu_ps,
                state_steps,
            )

            if len(params_with_grad) == 0:
                continue

            torch._foreach_lerp_(momentum_buffers, grads, 1 - group['momentum'])

            preconditioned_grads = []
            for p, g, merged_shape, Q, L, diag, mu_p_size in zip(
                params_with_grad, momentum_buffers, merged_shapes,
                Qs, Ls, diags, mu_ps
            ):
                state = self.state[p]
                
                state["step"] += 1
                
                original_shape = g.shape
                g_reshaped = g.view(merged_shape)

                if g_reshaped.ndim <= 1:
                    g_preconditioned = update_diag_solo(
                        Q[0], L[0], g_reshaped, group["preconditioner_lr"], state["step"]
                    )
                else:
                    if state["step"] % 100 == 0:
                        ql, qr = Q[0], Q[1]
                        max_l = ql.abs().max()
                        max_r = qr.abs().max()
                        rho = (max_l / max_r).sqrt()
                        Q[0] /= rho
                        Q[1] *= rho
                    
                    term2_target = mu_p_size if group["lr_style"] == "mu-p" else g_reshaped.numel()

                    if not diag[0] and not diag[1]:
                        g_preconditioned = precondition_DD(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            term2_target=term2_target
                        )
                    elif diag[0] and not diag[1]:
                        g_preconditioned = precondition_dD(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            term2_target=term2_target
                        )
                    elif not diag[0] and diag[1]:
                        g_preconditioned = precondition_Dd(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            term2_target=term2_target
                        )
                    else:
                        g_preconditioned = precondition_dd(
                            *Q, *L,
                            G=g_reshaped,
                            precond_lr=group["preconditioner_lr"],
                            step=state["step"],
                            term2_target=term2_target
                        )

                original_shape = p.grad.shape
                if original_shape != g_preconditioned.shape:
                    g_preconditioned = g_preconditioned.view(original_shape)
                
                preconditioned_grads.append(trust_region(g_preconditioned).to(dtype=p.dtype))
            
            if group["weight_decay"] > 0:
                torch._foreach_mul_(params_with_grad, 1 - group["lr"] * group["weight_decay"])
            
            torch._foreach_add_(
                params_with_grad,
                preconditioned_grads,
                alpha=-group["lr"] / 5.0 if group["lr_style"] == "adam" else -group["lr"]
            )

        return loss


BETA_L = 0.95
QUAD4P = False


def get_precond_lr(lr, step):
    return torch.clamp(lr * torch.rsqrt(1.0 + step / 10000.0), min=0.1)


@torch.compile(fullgraph=True)
def update_diag_solo(Q, L, G, precond_lr, step):
    Pg = (Q if QUAD4P else Q * Q) * G
    term1 = Pg * Pg
    term2 = 1.0
    ell = torch.amax(term1) + term2
    L.copy_(torch.max(BETA_L * L + (1 - BETA_L) * ell, ell))
    lr_over_2L = get_precond_lr(precond_lr, step) / (L if QUAD4P else 2 * L)
    gain = 1 - lr_over_2L * (term1 - term2)
    Q.mul_(gain * gain)
    return Pg


def _diag_update(term1, term2, L, Q, precond_lr, step):
    ell = torch.amax(term1) + term2
    L.copy_(torch.maximum(BETA_L * L + (1 - BETA_L) * ell, ell))
    lr_over_2L = get_precond_lr(precond_lr, step) / (L if QUAD4P else 2 * L)
    gain = 1 - lr_over_2L * (term1 - term2)
    Q.mul_(gain * gain)


def lb(A_outer: torch.Tensor):
    max_abs = A_outer.diagonal().max()

    def _inner():
        A = A_outer / max_abs
        j = torch.argmax(torch.sum(A * A, dim=1))
        x = A.index_select(0, j).view(-1)
        x = A.mv(x).float()
        x = x / x.norm()
        return (A.mv(x.to(A.dtype))).norm() * max_abs.squeeze().clone()

    return torch.cond(max_abs > 0, _inner, lambda: max_abs.squeeze().clone())


def _dense_update(term1, term2, L, Q, precond_lr, step):
    ell = lb(term1) + term2
    L.copy_(torch.maximum(BETA_L * L + (1 - BETA_L) * ell, ell))
    lr_over_2L = get_precond_lr(precond_lr, step) / (L if QUAD4P else 2 * L)
    # original
    # p = Q - lr_over_2L * (term1 @ Q - term2 * Q)
    # p = p - lr_over_2L * (p @ term1 - p * term2)
    # multiplicative
    scale1 = 1 + lr_over_2L * term2
    p = scale1 * Q - lr_over_2L * (term1 @ Q)
    p = scale1 * p - lr_over_2L * (p @ term1)
    # matmul
    # M = -lr_over_2L * term1
    # M.diagonal().add_(1 + lr_over_2L * term2)
    # p = M @ Q @ M
    Q.copy_((p + p.T) / 2)


@torch.compile(fullgraph=True)
def precondition_dd(Ql, Qr, Ll, Lr, G, precond_lr, step, term2_target):
    """Diagonal-diagonal preconditioning."""
    Pg = (Ql if QUAD4P else Ql * Ql).unsqueeze(1) * G * (Qr if QUAD4P else Qr * Qr).unsqueeze(0)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = term2_target / Ql.shape[0]
    _diag_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = term2_target / Qr.shape[0]
    _diag_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return Pg


@torch.compile(fullgraph=True)
def precondition_dD(Ql, Qr, Ll, Lr, G, precond_lr, step, term2_target):
    """Diagonal-dense preconditioning."""
    Pg = (Ql if QUAD4P else Ql * Ql).unsqueeze(1) * G @ (Qr if QUAD4P else Qr.T @ Qr)
    
    # left diagonal update
    term1_l = (Pg * Pg).sum(1)
    term2_l = term2_target / Ql.shape[0]
    _diag_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = term2_target / Qr.shape[0]
    _dense_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return Pg


@torch.compile(fullgraph=True)
def precondition_Dd(Ql, Qr, Ll, Lr, G, precond_lr, step, term2_target):
    """Dense-diagonal preconditioning."""
    Pg = (Ql if QUAD4P else Ql.T @ Ql) @ G * (Qr if QUAD4P else Qr * Qr).unsqueeze(0)
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = term2_target / Ql.shape[0]
    _dense_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right diagonal update
    term1_r = (Pg * Pg).sum(0)
    term2_r = term2_target / Qr.shape[0]
    _diag_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return Pg


@torch.compile(fullgraph=True)
def precondition_DD(Ql, Qr, Ll, Lr, G, precond_lr, step, term2_target):
    """Dense-dense preconditioning."""
    Pg = (Ql if QUAD4P else Ql.T @ Ql) @ G @ (Qr if QUAD4P else Qr.T @ Qr)
    
    # left dense update
    term1_l = Pg @ Pg.T
    term2_l = term2_target / Ql.shape[0]
    _dense_update(term1_l, term2_l, Ll, Ql, precond_lr, step)
    
    # right dense update
    term1_r = Pg.T @ Pg
    term2_r = term2_target / Qr.shape[0]
    _dense_update(term1_r, term2_r, Lr, Qr, precond_lr, step)
    
    return Pg


@torch.compile(fullgraph=True)
def trust_region(x):
    # tails too aggressive at scale, soft clip
    # TODO add option to not use this all the time
    return torch.tanh(x / 3.0) * 3.0


def merge_dims(tensor):
    if tensor.ndim <= 2:
        return tensor.shape
    dims = list(tensor.shape)
    best_ratio = float('inf')
    best_split = 1
    for split_idx in range(1, len(dims)):
        left_prod = math.prod(dims[:split_idx])
        right_prod = math.prod(dims[split_idx:])
        ratio = max(left_prod, right_prod) / min(left_prod, right_prod)
        if ratio < best_ratio:
            best_ratio = ratio
            best_split = split_idx
    return math.prod(dims[:best_split]), math.prod(dims[best_split:])


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    import time
    from dataclasses import dataclass, asdict
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    
    try:
        import wandb
        WANDB_AVAILABLE = True
    except ImportError:
        WANDB_AVAILABLE = False
        print("Warning: wandb not installed. Install with: pip install wandb")

    @dataclass
    class Config:
        # General settings
        seed: int = 42
        device_id: int = 0
        num_epochs: int = 100
        batch_size: int = 256
        num_workers: int = 4
        
        # Optimizer settings
        lr: float = 0.0003
        lr_style: str = "adam"
        momentum: float = 0.95
        weight_decay: float = 0.3
        preconditioner_lr: float = 0.8
        max_size_dense: int = 8192
        max_skew_dense: float = 1.0
        normalize_grads: bool = False
        dtype: str = "bf16"
        
        # Learning rate schedule
        warmup_steps: int = 100
        min_lr_ratio: float = 0.1  # decay to 0.1x original lr
        
        # Dataset settings
        dataset_name: str = "CIFAR-100"
        num_classes: int = 100
        image_size: int = 32
        
        # Training settings
        dropout_rate: float = 0.2
        log_interval: int = 10
        layer_norm_interval: int = 50
        print_interval: int = 100
        
        # WandB settings
        wandb_entity: str = "evanatyourservice"
        wandb_project: str = "quad-cifar"
        wandb_name: str = "quad-cifar100-baseline"
        
        # Additional info (not configurable)
        optimizer: str = "QUAD"
    
    cfg = Config()
    
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            config=asdict(cfg)
        )
        
        # Define custom x-axis for cleaner visualization
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("norms/*", step_metric="epoch")
        wandb.define_metric("update_energy/*", step_metric="epoch")
        wandb.define_metric("grad_energy/*", step_metric="epoch")
        wandb.define_metric("update_grad_ratio/*", step_metric="epoch")
        wandb.define_metric("update_param_ratio/*", step_metric="epoch")
        wandb.define_metric("precond_norm/*", step_metric="epoch")
        wandb.define_metric("param_norm/*", step_metric="epoch")
        wandb.define_metric("grad_norm/*", step_metric="epoch")

    class ConvNet(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
            self.ln1 = nn.LayerNorm([128, 32, 32])
            self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
            self.ln2 = nn.LayerNorm([256, 16, 16])
            self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
            self.ln3 = nn.LayerNorm([512, 8, 8])
            
            self.act = nn.GELU()
            self.pool = nn.MaxPool2d(2)
            self.dropout = nn.Dropout(cfg.dropout_rate)
            
            self.fc1 = nn.Linear(512 * 4 * 4, 1024, bias=False)
            self.ln4 = nn.LayerNorm(1024)
            self.fc2 = nn.Linear(1024, num_classes, bias=False)

        def forward(self, x):
            x = self.conv1(x)
            x = self.ln1(x)
            x = self.act(x)
            x = self.pool(x)
            
            x = self.conv2(x)
            x = self.ln2(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.pool(x)
            
            x = self.conv3(x)
            x = self.ln3(x)
            x = self.act(x)
            x = self.dropout(x)
            x = self.pool(x)
            
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.ln4(x)
            x = self.act(x)
            x = self.dropout(x)
            
            x = self.fc2(x)
            return x

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                            num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
                            prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
                           num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True,
                           prefetch_factor=2)

    model = ConvNet(num_classes=cfg.num_classes).to(device)
    model = torch.compile(model)

    opt_dtype = torch.bfloat16 if cfg.dtype in ["bfloat16", "bf16"] else torch.float32
    print(f"Using {opt_dtype} for optimizer")
    optimizer = QUAD(
        model.parameters(),
        lr=cfg.lr,
        lr_style=cfg.lr_style,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        preconditioner_lr=cfg.preconditioner_lr,
        max_size_dense=cfg.max_size_dense,
        max_skew_dense=cfg.max_skew_dense,
        normalize_grads=cfg.normalize_grads,
        dtype=opt_dtype,
    )

    steps_per_epoch = len(train_loader)
    total_steps = cfg.num_epochs * steps_per_epoch
    
    def get_lr(epoch, step):
        global_step = epoch * steps_per_epoch + step
        
        if global_step < cfg.warmup_steps:
            return cfg.lr * (global_step + 1) / cfg.warmup_steps
        else:
            # Linear decay from lr to min_lr_ratio * lr
            progress = (global_step - cfg.warmup_steps) / (total_steps - cfg.warmup_steps)
            return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 - progress))

    def compute_norms(model, compute_layer_norms=True):
        """Compute parameter and gradient norms efficiently."""
        with torch.no_grad():
            param_norms = []
            grad_norms = []
            layer_norms = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    p_norm = param.data.norm(2)
                    g_norm = param.grad.data.norm(2)
                    
                    param_norms.append(p_norm)
                    grad_norms.append(g_norm)
                    
                    if compute_layer_norms:
                        layer_norms[f"param_norm/{name}"] = p_norm.item()
                        layer_norms[f"grad_norm/{name}"] = g_norm.item()
            
            # Compute total norms efficiently
            if param_norms:
                param_norm = torch.stack(param_norms).norm(2).item()
                grad_norm = torch.stack(grad_norms).norm(2).item()
            else:
                param_norm = 0.0
                grad_norm = 0.0
        
        return param_norm, grad_norm, layer_norms

    def train_epoch(model, loader, optimizer, device, epoch):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        # Calculate global step offset for this epoch
        global_step_offset = epoch * len(loader)
        
        for batch_idx, (data, target) in enumerate(loader):
            current_lr = get_lr(epoch, batch_idx)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            
            # Inject zero gradient at global step 10
            global_step = global_step_offset + batch_idx
            if global_step == 10:
                # Zero out gradient for conv1.weight
                if hasattr(model, 'conv1') and model.conv1.weight.grad is not None:
                    print(f"\n!!! Injecting ZERO gradient at step {global_step} for conv1.weight !!!")
                    print(f"Original grad norm: {model.conv1.weight.grad.norm().item():.6f}")
                    model.conv1.weight.grad.zero_()
                    print(f"After zeroing grad norm: {model.conv1.weight.grad.norm().item():.6f}\n")
            
            # Compute norms before optimizer step
            compute_layer_norms = WANDB_AVAILABLE and batch_idx % cfg.layer_norm_interval == 0
            param_norm, grad_norm, layer_norms = compute_norms(model, compute_layer_norms)
            
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Log to wandb
            if WANDB_AVAILABLE and batch_idx % cfg.log_interval == 0:                
                log_dict = {
                    "train/loss": loss.item(),
                    "train/accuracy": 100. * correct / total,
                    "train/lr": current_lr,
                    "norms/param_norm": param_norm,
                    "norms/grad_norm": grad_norm,
                    "epoch": epoch,
                }
                # Only log layer norms every layer_norm_interval batches
                if batch_idx % cfg.layer_norm_interval == 0:
                    log_dict.update(layer_norms)
                wandb.log(log_dict)
            
            if batch_idx % cfg.print_interval == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(loader)}] Loss: {loss.item():.4f} '
                      f'LR: {current_lr:.6f}')
        
        return total_loss / total, 100. * correct / total

    def evaluate(model, loader, device):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        return total_loss / total, 100. * correct / total

    total_params = sum(p.numel() for p in model.parameters())
    print(f"CIFAR-100 Classification with QUAD Optimizer")
    print(f"Model parameters: {total_params:,}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr} (warmup: {cfg.warmup_steps} steps, decay to {cfg.min_lr_ratio}x)")
    print(f"Epochs: {cfg.num_epochs}")
    print("-" * 60)

    if WANDB_AVAILABLE:
        wandb.config.update({"total_params": total_params})

    best_acc = 0
    for epoch in range(cfg.num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, device)
        epoch_time = time.time() - start_time
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Log epoch metrics
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_accuracy": train_acc,
                "val/loss": test_loss,
                "val/accuracy": test_acc,
                "val/best_accuracy": best_acc,
                "time/epoch": epoch_time,
            })
        
        print(f'Epoch {epoch+1}/{cfg.num_epochs}: {epoch_time:.1f}s, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% (Best: {best_acc:.2f}%)')

    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    
    if WANDB_AVAILABLE:
        final_lr = get_lr(cfg.num_epochs-1, steps_per_epoch-1)
        
        wandb.log({
            "final/test_accuracy": test_acc,
            "final/best_accuracy": best_acc,
        })
        
        # Log model summary
        wandb.summary["final_test_accuracy"] = test_acc
        wandb.summary["best_test_accuracy"] = best_acc
        wandb.summary["total_parameters"] = total_params
        wandb.summary["final_learning_rate"] = final_lr
        
        wandb.finish()
