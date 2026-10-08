"""Training and evaluation of one epoch on the OGB molecular datasets."""

import torch

from train.pe_input import get_batch_pos_enc


def _perf(perf, dataset):
    if dataset == "ogbg-molpcba":
        return perf['ap']
    if dataset == "ogbg-moltox21":
        return perf['rocauc']
    raise ValueError("unsupported OGB dataset %r" % dataset)


def train_epoch_sparse(model, optimizer, device, data_loader, epoch, evaluator, dataset):
    model.train()
    epoch_loss = 0
    y_true = []
    y_pred = []
    for iter, (batch_graphs, batch_labels, batch_snorm_n) in enumerate(data_loader):
        optimizer.zero_grad()
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        batch_pos_enc = get_batch_pos_enc(model, batch_graphs, device, training=True)
        batch_pred = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
        # ignore nan labels (unlabeled) when computing training loss
        is_labeled = batch_labels == batch_labels
        loss = model.loss(batch_pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
        loss.backward()
        optimizer.step()
        y_true.append(batch_labels.view(batch_pred.shape).detach().cpu())
        y_pred.append(batch_pred.detach().cpu())
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    # compute performance metric using the OGB evaluator
    perf = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return epoch_loss, _perf(perf, dataset), optimizer


def evaluate_network_sparse(model, device, data_loader, epoch, evaluator, dataset):
    model.eval()
    epoch_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_labels, batch_snorm_n) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_pos_enc = get_batch_pos_enc(model, batch_graphs, device, training=False)
            batch_pred = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
            # ignore nan labels (unlabeled) when computing loss
            is_labeled = batch_labels == batch_labels
            loss = model.loss(batch_pred.to(torch.float32)[is_labeled], batch_labels.to(torch.float32)[is_labeled])
            y_true.append(batch_labels.view(batch_pred.shape).detach().cpu())
            y_pred.append(batch_pred.detach().cpu())
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    perf = evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return epoch_loss, _perf(perf, dataset)
