"""Graph transformer on OGB molecule classification (MolHIV, MolClinTox, MolBBBP).

Adapted from https://github.com/luis-mueller/wl-transformers (MIT License,
Copyright (c) 2024 Luis Mueller).  Laplacian eigenvectors are canonicalized with
the ``prism`` package (PRiSM, MAP, OAP) or randomly sign flipped (LPE); see
``wl_transformers.pe_types``.
"""

import hydra
import torch
import torch.nn.functional as F
from loguru import logger
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
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
)


@torch.no_grad()
def evaluation(evaluator, model, loader, device, tdtype, metric, log_per_task=False):
    y_pred = []
    y_true = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
            logits = model(batch)
        y_true.append(batch.y.view(logits.shape).detach().cpu())
        y_pred.append(logits.detach().cpu().float())  # exact upcast (bfloat16 on CPU)
    y_true_cat = torch.cat(y_true)
    y_pred_cat = torch.cat(y_pred)
    result = evaluator.eval(dict(y_true=y_true_cat, y_pred=y_pred_cat))[metric]
    if log_per_task and y_true_cat.dim() == 2 and y_true_cat.size(1) > 1:
        import numpy as np
        from sklearn.metrics import roc_auc_score

        per_task = []
        for t in range(y_true_cat.size(1)):
            mask = ~torch.isnan(y_true_cat[:, t])
            if mask.sum() > 0:
                try:
                    auc = roc_auc_score(
                        y_true_cat[mask, t].numpy(),
                        y_pred_cat[mask, t].numpy(),
                    )
                    per_task.append(auc)
                except ValueError:
                    per_task.append(float("nan"))
            else:
                per_task.append(float("nan"))
        logger.info(f"Per-task AUC: {[f'{a:.4f}' for a in per_task]}")
        logger.info(f"Per-task mean: {np.nanmean(per_task):.4f}, std: {np.nanstd(per_task):.4f}")
    return result


@hydra.main(version_base=None, config_path="./configs", config_name="ogbg")
def main(cfg):
    data_dir = ensure_data_dir(cfg.root)
    run_logger = RunLogger(cfg, config_dict=dict(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Accelerator: {device}")

    tdtype = autocast_dtype(device)
    logger.info(f"Autocast data type: {tdtype}")

    seed_everything(cfg.seed)
    logger.info(f"Random seed: {cfg.seed}")

    refine = bool(cfg.refine_ordering)
    canonicalizer = make_canonicalizer(cfg.pe_type, refine, cfg.prism_preset)
    log_config(cfg, pe_type=cfg.pe_type, canonicalizer=canonicalizer, refine_ordering=refine)

    transform = Transform(
        cfg.num_vecs, cfg.order, cfg.normalized_laplacian, cfg.normalize_eigenvecs,
        canonicalizer=canonicalizer,
    )

    dataset = PygGraphPropPredDataset(
        name=cfg.dataset_name,
        root=data_dir,
        transform=transform,
    )
    logger.info(f"Dataset {cfg.dataset_name} has {len(dataset)} molecules")

    split_idx = dataset.get_idx_split()
    if cfg.smoke:
        split_idx = {k: v[: cfg.smoke_graphs] for k, v in split_idx.items()}
    num_epochs = 1 if cfg.smoke else cfg.num_epochs
    train_loader = DataLoader(
        dataset[split_idx["train"]], batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        dataset[split_idx["valid"]], batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        dataset[split_idx["test"]], batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers,
    )

    evaluator = Evaluator(cfg.dataset_name)
    metric = evaluator.eval_metric
    assert metric in ("rocauc", "ap", "rmse"), f"Unsupported metric: {metric}"
    higher_is_better = metric in ("rocauc", "ap")

    val_after = len(dataset[split_idx["train"]]) // cfg.batch_size
    assert val_after > 0, "training split smaller than one batch"
    log_after = max(1, val_after // 10)
    num_steps = val_after * num_epochs

    atom_encoder = AtomEncoder(emb_dim=cfg.embed_dim)
    bond_encoder = BondEncoder(emb_dim=cfg.embed_dim)

    logger.info("Creating transformer")
    model = create_transformer(
        cfg, atom_encoder, bond_encoder, output_dim=dataset.num_tasks, device=device
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
        warmup_epochs * val_after,
        lr=cfg.lr,
        lr_decay_iters=num_steps,
        min_lr=0,
    )
    logger.info(f"Cosine lr scheduler with {warmup_epochs} warm-up epochs ready")

    scaler = torch.cuda.amp.GradScaler(enabled=(tdtype == torch.float16))
    logger.info(
        f"Logging after {log_after}, validating after {val_after} steps. Training for {num_steps} steps"
    )

    step = 1
    best_val_score = None
    test_score = None
    while step <= num_steps:
        model.train()
        loss_window = []
        for batch in train_loader:
            batch = batch.to(device)
            with torch.autocast(device_type=device, dtype=tdtype, enabled=True):
                logits = model(batch)
                is_labeled = batch.y == batch.y
                if higher_is_better:
                    loss = F.binary_cross_entropy_with_logits(
                        logits[is_labeled], batch.y.to(tdtype)[is_labeled]
                    )
                else:
                    loss = F.mse_loss(
                        logits[is_labeled], batch.y.to(tdtype)[is_labeled]
                    )
            scaler.scale(loss).backward()
            if cfg.gradient_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.gradient_norm
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if step % int(log_after) == 0:
                if len(loss_window) == 0:
                    logger.warning("No loss data available")
                else:
                    run_logger.log(
                        dict(
                            train_loss=sum(loss_window) / len(loss_window),
                            lr=optimizer.param_groups[0]["lr"],
                        ),
                        step=step,
                    )
                loss_window = []
            else:
                loss_window.append(float(loss.detach().cpu()))

            if step % int(val_after) == 0:
                logger.info("Evaluating model")
                model.eval()
                val_score = evaluation(
                    evaluator, model, val_loader, device, tdtype, metric
                )
                improved = (best_val_score is None or
                            (val_score > best_val_score if higher_is_better
                             else val_score < best_val_score))
                if improved:
                    best_val_score = val_score
                    test_score = evaluation(
                        evaluator, model, test_loader, device, tdtype, metric,
                        log_per_task=True,
                    )

                logger.info(
                    dict(
                        step=step,
                        val_score=val_score,
                        best_val_score=best_val_score,
                        test_score=test_score,
                    )
                )
                run_logger.log(
                    dict(
                        val_score=val_score,
                        best_val_score=best_val_score,
                        test_score=test_score,
                    ),
                    step=step,
                )
                model.train()

            step += 1

    logger.info("Training complete")
    logger.info(f"Final | best_val_score={best_val_score} test_score={test_score}")
    run_logger.finish()


if __name__ == "__main__":
    main()
