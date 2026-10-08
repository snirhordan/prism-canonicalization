"""Optimizer, schedule and logging utilities shared by the entry points.

``configure_optimizers`` and ``CosineWithWarmupLR`` are adapted from nanoGPT
(https://github.com/karpathy/nanoGPT, MIT License, Copyright (c) 2022 Andrej
Karpathy); see ``THIRD_PARTY_NOTICES``.  Weights & Biases logging is optional
and off unless ``wandb_project`` is set.
"""

from __future__ import annotations

import inspect
import math
import os

import torch
from loguru import logger
from prettytable import PrettyTable


def configure_optimizers(model: torch.nn.Module, weight_decay, learning_rate, betas, device_type):
    """AdamW with weight decay on matrices only (adapted from nanoGPT)."""
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # Every parameter with two or more dimensions is weight decayed, biases and
    # layer norms are not.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {use_fused}")
    return optimizer


def autocast_dtype(device: str):
    """Mixed-precision dtype of a run: float16 on CUDA (as in all reported runs),
    bfloat16 on CPU (float16 autocast is unavailable there; smoke tests only)."""
    return torch.float16 if device == "cuda" else torch.bfloat16


def count_parameters(model: torch.nn.Module, verbose: bool = False) -> int:
    """Number of trainable parameters (optionally logs the per-tensor table)."""
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    if verbose:
        logger.info(f"\n{str(table)}")
    return total_params


def ensure_data_dir(root: str) -> str:
    """Create and return ``{root}/data``."""
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


class CosineWithWarmupLR:
    """Linear warm-up followed by cosine decay (adapted from nanoGPT)."""

    def __init__(self, optimizer, warmup_iters: int, lr: float, lr_decay_iters: int,
                 min_lr: float, epoch: int = -1):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.lr = lr
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.epoch = epoch
        self.step()

    def step(self):
        self.epoch += 1
        lr = self._get_lr(self.epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _get_lr(self, epoch: int):
        if epoch < self.warmup_iters:
            return self.lr * epoch / self.warmup_iters
        if epoch > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (epoch - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.lr - self.min_lr)


def log_config(cfg, *, pe_type, canonicalizer, refine_ordering):
    """Log the configuration that determines the canonicalization of a run."""
    from .pe_types import describe_canonicalizer

    logger.info(
        "Preflight | pe_type={pe} refine_ordering={r} lr={lr} batch_size={bs} "
        "num_epochs={ne} canonicalizer={c}",
        pe=pe_type,
        r=refine_ordering,
        lr=getattr(cfg, "lr", None),
        bs=getattr(cfg, "batch_size", None),
        ne=getattr(cfg, "num_epochs", None),
        c=describe_canonicalizer(canonicalizer),
    )


class RunLogger:
    """Optional Weights & Biases logger; a no-op unless ``wandb_project`` is set."""

    def __init__(self, cfg, config_dict=None):
        self.run = None
        project = getattr(cfg, "wandb_project", None)
        if project is None:
            return
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb_project is set but wandb is not installed (pip install wandb)"
            ) from exc
        self._wandb = wandb
        self.run = wandb.init(
            project=project,
            entity=getattr(cfg, "wandb_entity", None),
            name=getattr(cfg, "wandb_name", None),
            dir=getattr(cfg, "root", "."),
            config=config_dict,
        )

    def log(self, metrics: dict, step=None):
        if self.run is not None:
            self._wandb.log(metrics, step=step)

    def finish(self):
        if self.run is not None:
            self._wandb.finish()
