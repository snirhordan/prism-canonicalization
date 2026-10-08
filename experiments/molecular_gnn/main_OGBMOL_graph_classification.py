"""Graph classification on OGB MolTox21 (ROC-AUC) and MolPCBA (average precision) with GatedGCN or PNA.

    python main_OGBMOL_graph_classification.py --config configs/PNA_MOLPCBA_PRiSM.json [--seed 41]

Without ``--seed`` the four seeds 41, 95, 12, 35 run one after the other and the mean and standard
deviation (``numpy.std``, ddof = 0) of the test metric at the epoch of best validation metric are
printed.  See README.md.

Based on the harness of Ma et al. (PKU-ML/canonicalization, MIT License), which builds on
SignNet (Lim et al.), gnn-lspe and benchmarking-gnns (Dwivedi et al.).
"""

import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float64)  # pre-process with double, train with float

from graphdata.data import LoadData  # noqa: E402
from graphdata.positional_encoding import add_pos_enc  # noqa: E402
from nets.OGBMOL_graph_classification.load_net import gnn_model  # noqa: E402
from train.train_OGBMOL_graph_classification import evaluate_network_sparse as evaluate_network  # noqa: E402
from train.train_OGBMOL_graph_classification import train_epoch_sparse as train_epoch  # noqa: E402
from utils.cli import SMOKE_SIZES, build_parser, load_config  # noqa: E402
from utils.runtime import count_parameters, gpu_setup, run_paths, seed_everything  # noqa: E402

METRIC_NAMES = {'ogbg-moltox21': ('AUC', 'TEST AUC', 'TRAIN AUC', 'VAL AUC'),
                'ogbg-molpcba': ('AP', 'TEST AVG PRECISION', 'TRAIN AVG PRECISION', 'VAL AVG PRECISION')}


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, out_dir, num_workers):
    t0 = time.time()
    per_epoch_time = []
    DATASET_NAME = dataset.name
    short, test_key, train_key, val_key = METRIC_NAMES[DATASET_NAME]
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    evaluator = dataset.evaluator
    device = net_params['device']
    write_file_name, write_config_file = run_paths(out_dir, MODEL_NAME, DATASET_NAME, net_params['gpu_id'],
                                                   params['seed'])
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, {k: v for k, v in net_params.items() if k != 'avg_d'},
            net_params['total_param']))
    seed_everything(params['seed'], device)
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))
    torch.set_default_dtype(torch.float)  # pre-process with double, train with float
    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'])
    epoch_train_accs, epoch_val_accs, epoch_test_accs = [], [], []
    best_val_acc = -float('inf')
    early_stop_counter = 0
    early_stop_patience = params.get('early_stopping_patience', 0)  # 0 = disabled
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(trainset, num_workers=num_workers, batch_size=params['batch_size'], shuffle=True,
                              collate_fn=dataset.collate, pin_memory=pin_memory)
    val_loader = DataLoader(valset, num_workers=num_workers, batch_size=params['batch_size'], shuffle=False,
                            collate_fn=dataset.collate, pin_memory=pin_memory)
    test_loader = DataLoader(testset, num_workers=num_workers, batch_size=params['batch_size'], shuffle=False,
                             collate_fn=dataset.collate, pin_memory=pin_memory)
    # At any point you can hit Ctrl + C to break out of training early.
    epoch = 0
    try:
        for epoch in range(params['epochs']):
            start = time.time()
            epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                       epoch, evaluator, dataset.name)
            epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch, evaluator,
                                                             dataset.name)
            _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch, evaluator, dataset.name)
            epoch_train_accs.append(epoch_train_acc)
            epoch_val_accs.append(epoch_val_acc)
            epoch_test_accs.append(epoch_test_acc)
            per_epoch_time.append(time.time() - start)
            print('Epoch %d: time=%.1fs lr=%.2e train_loss=%.4f val_loss=%.4f train_%s=%.4f val_%s=%.4f '
                  'test_%s=%.4f' % (epoch, per_epoch_time[-1], optimizer.param_groups[0]['lr'], epoch_train_loss,
                                    epoch_val_loss, short, epoch_train_acc, short, epoch_val_acc, short,
                                    epoch_test_acc), flush=True)
            scheduler.step(epoch_val_loss)
            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("!! LR EQUAL TO MIN LR SET.")
                break
            # Early stopping on the validation metric
            if early_stop_patience > 0:
                if epoch_val_acc > best_val_acc:
                    best_val_acc = epoch_val_acc
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"!! Early stopping: val acc not improved for {early_stop_patience} epochs.")
                    break
            # Stop training after params['max_time'] hours
            if time.time() - t0 > params['max_time'] * 3600:
                print('-' * 89)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    # OGB: test score at the epoch of best validation score
    epoch_best = epoch_val_accs.index(max(epoch_val_accs))
    test_acc = epoch_test_accs[epoch_best]
    train_acc = epoch_train_accs[epoch_best]
    val_acc = epoch_val_accs[epoch_best]
    print("Test {}: {:.4f}".format(short, test_acc))
    print("Train {}: {:.4f}".format(short, train_acc))
    print("Val {}: {:.4f}".format(short, val_acc))
    print("Best epoch: {}".format(epoch_best))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    # Write the results in out_dir/results
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\n{}: {:.4f}\n{}: {:.4f}\n{}: {:.4f}\n\n
    Best epoch: {}\nConvergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params['total_param'],
                        test_key, test_acc, train_key, train_acc, val_key, val_acc,
                        epoch_best, epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))
    return test_acc


def main():
    args = build_parser('ogb').parse_args()
    config, params, net_params, seeds = load_config(args, 'ogb')
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    MODEL_NAME = config['model']
    DATASET_NAME = config['dataset']
    out_dir = config['out_dir']
    dataset = LoadData(DATASET_NAME, args.data_root)
    if args.smoke:
        dataset.subset(*SMOKE_SIZES)
    # network parameters
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    net_params['n_classes'] = dataset.dataset.num_tasks  # provided by the OGB dataset class
    if MODEL_NAME == 'PNA':
        D = torch.cat([g.in_degrees().float() for g in dataset.train.graph_lists])
        net_params['avg_d'] = dict(lin=torch.mean(D),
                                   exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                                   log=torch.mean(torch.log(D + 1)))
    os.makedirs(out_dir, exist_ok=True)
    net_params['total_param'] = count_parameters(gnn_model(MODEL_NAME, net_params))
    print('MODEL/Total parameters:', MODEL_NAME, net_params['total_param'])
    # positional encoding (float64 numerics, cached)
    add_pos_enc(dataset, net_params['pe_init'], net_params.get('prism_preset'), net_params['pos_enc_dim'],
                cache_dir=args.cache_dir, workers=args.pe_workers, use_cache=not args.smoke)
    if args.build_pe_only:
        return
    accs = []
    for seed in seeds:
        params['seed'] = seed
        accs.append(train_val_pipeline(MODEL_NAME, dataset, params, net_params, out_dir, args.num_workers))
        print()
    print("Final accs:", accs)
    print("Mean:", np.mean(accs))
    print("Std:", np.std(accs))
    with open(os.path.join(out_dir, 'results', 'final_%s.json' % time.strftime('%Y%m%d_%H%M%S')), 'w') as f:
        json.dump({'config': args.config, 'seeds': list(seeds), 'test': accs,
                   'mean': float(np.mean(accs)), 'std': float(np.std(accs))}, f, indent=1)


if __name__ == '__main__':
    main()
