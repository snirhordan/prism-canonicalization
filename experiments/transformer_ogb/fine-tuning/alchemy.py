"""Graph transformer on Alchemy (12-task quantum-chemistry regression, 10K subset).

Adapted from https://github.com/luis-mueller/wl-transformers (MIT License,
Copyright (c) 2024 Luis Mueller).  Laplacian eigenvectors are canonicalized with
the ``prism`` package (PRiSM, MAP, OAP) or randomly sign flipped (LPE); see
``wl_transformers.pe_types``.

The 10K subset (10000 train, 1000 validation, 1000 test graphs of the public
``alchemy_full`` dataset) is given by the ``*_al_10.index`` files next to this
script.
"""

import os

import hydra
import torch
from loguru import logger
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.seed import seed_everything

from wl_transformers import (
    CosineWithWarmupLR,
    RunLogger,
    Transform,
    autocast_dtype,
    configure_optimizers,
    create_transformer,
    ensure_data_dir,
    log_config,
    make_canonicalizer,
    transform_dataset,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def read_indices(name):
    with open(os.path.join(HERE, f"{name}_al_10.index")) as fh:
        return [int(i) for i in fh.read().strip().split(",")]


def load_dataset(root: str):
    indices = read_indices("train") + read_indices("val") + read_indices("test")
    return TUDataset(f"{root}/alchemy", name="alchemy_full")[indices]


@hydra.main(version_base=None, config_path="configs", config_name="alchemy")
def main(cfg):
    data_dir = ensure_data_dir(cfg.root)
    run_logger = RunLogger(cfg, config_dict=dict(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed}")

    refine = bool(cfg.refine_ordering)
    canonicalizer = make_canonicalizer(cfg.pe_type, refine, cfg.prism_preset)
    log_config(cfg, pe_type=cfg.pe_type, canonicalizer=canonicalizer, refine_ordering=refine)

    dataset = load_dataset(data_dir)
    if cfg.smoke:
        n = cfg.smoke_graphs
        dataset = dataset[list(range(n)) + list(range(10000, 10000 + n))
                          + list(range(11000, 11000 + n))]

    transform = Transform(
        cfg.num_vecs, cfg.order, cfg.normalized_laplacian, cfg.normalize_eigenvecs,
        canonicalizer=canonicalizer,
    )

    logger.info("Pre-transforming dataset")
    dataset = transform_dataset(dataset, transform)

    mean = dataset.y.mean(dim=0, keepdim=True)
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    if cfg.smoke:
        n = cfg.smoke_graphs
        train_dataset = dataset[0:n]
        val_dataset = dataset[n:2 * n]
        test_dataset = dataset[2 * n:]
    else:
        train_dataset = dataset[0:10000]
        val_dataset = dataset[10000:11000]
        test_dataset = dataset[11000:]
    num_epochs = 1 if cfg.smoke else cfg.num_epochs

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.num_workers)

    atom_encoder = torch.nn.Linear(6, cfg.embed_dim)
    bond_encoder = torch.nn.Linear(4, cfg.embed_dim)

    logger.info("Creating transformer")
    model = create_transformer(
        cfg, atom_encoder, bond_encoder, output_dim=12, device=device
    )
    logger.info(model)

    optimizer = configure_optimizers(
        model,
        cfg.weight_decay,
        cfg.lr,
        (0.9, 0.95),
        device,
    )
    logger.info(f"Optimizer with lr {cfg.lr}, weight decay {cfg.weight_decay} ready")

    warmup_epochs = int(0.025 * num_epochs)
    scheduler = CosineWithWarmupLR(
        optimizer,
        warmup_epochs,
        lr=cfg.lr,
        lr_decay_iters=num_epochs,
        min_lr=0,
    )
    logger.info(f"Cosine lr scheduler with {warmup_epochs} warm-up epochs ready")

    amp_dtype = autocast_dtype(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    def train():
        model.train()
        loss_all = 0

        lf = torch.nn.L1Loss()
        for data in train_loader:
            data = data.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=True):
                loss = lf(model(data), data.y)
            scaler.scale(loss).backward()
            if cfg.gradient_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.gradient_norm
                )
            loss_all += loss.item() * data.num_graphs
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        return loss_all / len(train_loader.dataset)

    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=True):
                error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)
        return error.mean().item()

    best_val_error = None
    test_error = None
    logger.info(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        scheduler.step()
        lr = scheduler.optimizer.param_groups[0]["lr"]
        loss = train()
        val_error = test(val_loader)

        if best_val_error is None or val_error <= best_val_error:
            test_error = test(test_loader)
            best_val_error = val_error

        run_logger.log(
            {
                "lr": lr,
                "loss": loss,
                "val_error": val_error,
                "test_error": test_error,
            }
        )

        logger.info(
            f"Epoch: {epoch} LR: {lr:.5f} Loss: {loss:.5f} Val. error {val_error:.5f} Test error {test_error:.5f}"
        )

    logger.info("Training complete")
    logger.info(f"Final | best_val_error={best_val_error} test_error={test_error}")
    run_logger.finish()


if __name__ == "__main__":
    main()
