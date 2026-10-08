"""Training and evaluation of one epoch on ZINC."""

import torch

from train.metrics import MAE
from train.pe_input import get_batch_pos_enc


def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    for iter, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        optimizer.zero_grad()
        batch_pos_enc = get_batch_pos_enc(model, batch_graphs, device, training=True)
        batch_scores = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    return epoch_loss, epoch_train_mae, optimizer


def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_pos_enc = get_batch_pos_enc(model, batch_graphs, device, training=False)
            batch_scores = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
    return epoch_test_loss, epoch_test_mae
