"""Graph regression on ZINC 12K with GatedGCN or PNA (test MAE).

    python main_ZINC_graph_regression.py --config configs/GatedGCN_ZINC_PRiSM.json [--seed 41]

Without ``--seed`` the four seeds 41, 95, 12, 35 run one after the other and the mean and standard
deviation (``numpy.std``, ddof = 0) of the test MAE are printed.  See README.md.

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
from nets.ZINC_graph_regression.load_net import gnn_model  # noqa: E402
from train.train_ZINC_graph_regression import evaluate_network_sparse as evaluate_network  # noqa: E402
from train.train_ZINC_graph_regression import train_epoch_sparse as train_epoch  # noqa: E402
from utils.cli import SMOKE_SIZES, build_parser, load_config  # noqa: E402
from utils.runtime import count_parameters, gpu_setup, run_paths, seed_everything  # noqa: E402


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, out_dir, num_workers):
    t0 = time.time()
    per_epoch_time = []
    DATASET_NAME = dataset.name
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
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
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = params.get('early_stopping_patience', 0)  # 0 = disabled
    train_loader = DataLoader(trainset, num_workers=num_workers, batch_size=params['batch_size'], shuffle=True,
                              collate_fn=dataset.collate)
    val_loader = DataLoader(valset, num_workers=num_workers, batch_size=params['batch_size'], shuffle=False,
                            collate_fn=dataset.collate)
    test_loader = DataLoader(testset, num_workers=num_workers, batch_size=params['batch_size'], shuffle=False,
                             collate_fn=dataset.collate)
    # At any point you can hit Ctrl + C to break out of training early.
    epoch = 0
    try:
        for epoch in range(params['epochs']):
            start = time.time()
            epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
            epoch_val_loss, epoch_val_mae = evaluate_network(model, device, val_loader, epoch)
            epoch_test_loss, epoch_test_mae = evaluate_network(model, device, test_loader, epoch)
            per_epoch_time.append(time.time() - start)
            print('Epoch %d: time=%.1fs lr=%.2e train_loss=%.4f val_loss=%.4f train_MAE=%.4f val_MAE=%.4f '
                  'test_MAE=%.4f' % (epoch, per_epoch_time[-1], optimizer.param_groups[0]['lr'], epoch_train_loss,
                                     epoch_val_loss, epoch_train_mae, epoch_val_mae, epoch_test_mae), flush=True)
            scheduler.step(epoch_val_loss)
            if optimizer.param_groups[0]['lr'] < params['min_lr']:
                print("!! LR EQUAL TO MIN LR SET.")
                break
            # Early stopping on val loss
            if early_stop_patience > 0:
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"!! Early stopping: val loss not improved for {early_stop_patience} epochs.")
                    break
            # Stop training after params['max_time'] hours
            if time.time() - t0 > params['max_time'] * 3600:
                print('-' * 89)
                print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    _, test_mae = evaluate_network(model, device, test_loader, epoch)
    _, train_mae = evaluate_network(model, device, train_loader, epoch)
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))
    # Write the results in out_dir/results
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params['total_param'],
                        test_mae, train_mae, epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))
    return test_mae


def main():
    args = build_parser('zinc').parse_args()
    config, params, net_params, seeds = load_config(args, 'zinc')
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
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type
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
    maes = []
    for seed in seeds:
        params['seed'] = seed
        maes.append(train_val_pipeline(MODEL_NAME, dataset, params, net_params, out_dir, args.num_workers))
        print()
    print("Final maes:", maes)
    print("Mean:", np.mean(maes))
    print("Std:", np.std(maes))
    with open(os.path.join(out_dir, 'results', 'final_%s.json' % time.strftime('%Y%m%d_%H%M%S')), 'w') as f:
        json.dump({'config': args.config, 'seeds': list(seeds), 'test_mae': maes,
                   'mean': float(np.mean(maes)), 'std': float(np.std(maes))}, f, indent=1)


if __name__ == '__main__':
    main()
