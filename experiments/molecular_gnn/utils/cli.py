"""Command line interface shared by the two entry points.

The config file (``configs/<Backbone>_<DATASET>_<method>.json``) holds ``params`` (training) and
``net_params`` (model, positional encoding).  Every field of the tables below can be overridden on
the command line.  ``--smoke`` runs one epoch on a few hundred graphs, to check an installation.
"""

import argparse
import json
import os

SMOKE_SIZES = (512, 128, 128)  # train / val / test graphs of a --smoke run
SEEDS = (41, 95, 12, 35)

# command line name -> (section of the config, config key, type)
_BOOL = lambda s: str(s).lower() in ('true', '1', 'yes')
_PARAMS = {
    'epochs': int, 'batch_size': int, 'init_lr': float, 'lr_reduce_factor': float,
    'lr_schedule_patience': int, 'min_lr': float, 'weight_decay': float, 'max_time': float,
}
_NET_PARAMS = {
    'L': int, 'hidden_dim': int, 'out_dim': int, 'residual': _BOOL, 'readout': str, 'dropout': float,
    'batch_norm': _BOOL, 'pos_enc_dim': int, 'pe_init': str, 'prism_preset': str, 'sign_inv_net': str,
    'sign_inv_layers': int, 'sign_inv_activation': str, 'phi_out_dim': int,
}


def build_parser(task):
    """``task`` is ``'zinc'`` or ``'ogb'``."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help="config file with training/model/data/param details")
    p.add_argument('--gpu_id', type=int, help="GPU index (ignored if CUDA_VISIBLE_DEVICES is already set)")
    p.add_argument('--model', help="GatedGCN or PNA")
    p.add_argument('--dataset', help="ZINC" if task == 'zinc' else "OGBG-MOLTOX21 or OGBG-MOLPCBA")
    p.add_argument('--out_dir', help="output directory (default: out_dir of the config)")
    p.add_argument('--seed', type=int, help="run this seed only (default: 41, 95, 12, 35 one after the other)")
    p.add_argument('--epochs', '--max_epochs', dest='epochs', help="maximum number of epochs")
    for name, typ in _PARAMS.items():
        if name != 'epochs':
            p.add_argument('--' + name, help="training parameter %s" % name)
    for name in _NET_PARAMS:
        p.add_argument('--' + name, help="model parameter %s" % name)
    if task == 'zinc':
        p.add_argument('--in_feat_dropout', help="model parameter in_feat_dropout")
        p.add_argument('--edge_feat', help="model parameter edge_feat")
    p.add_argument('--early_stopping_patience', type=int, default=0, help="early stopping patience (0 = disabled)")
    p.add_argument('--data_root', default=os.environ.get('PRISM_DATA_ROOT', 'data'),
                   help="where the datasets live (default: ./data or $PRISM_DATA_ROOT)")
    p.add_argument('--cache_dir', default=os.environ.get('PRISM_CACHE_DIR', 'cache'),
                   help="where positional encodings are cached (default: ./cache or $PRISM_CACHE_DIR)")
    p.add_argument('--num_workers', type=int, default=4, help="DataLoader workers")
    p.add_argument('--pe_workers', type=int, default=1,
                   help="processes used to compute the positional encodings (the result does not depend on it)")
    p.add_argument('--build_pe_only', action='store_true',
                   help="compute and cache the positional encodings, then exit")
    p.add_argument('--smoke', action='store_true',
                   help="1 epoch, seed 41, %d/%d/%d graphs, no PE cache, output under <out_dir>/smoke/" % SMOKE_SIZES)
    return p


def load_config(args, task):
    """Read the config and apply the command line overrides; returns ``(config, params, net_params)``."""
    with open(args.config) as f:
        config = json.load(f)
    params = config['params']
    net_params = config['net_params']
    for name, typ in _PARAMS.items():
        value = getattr(args, name, None)
        if value is not None:
            params[name] = typ(value)
    for name, typ in _NET_PARAMS.items():
        value = getattr(args, name, None)
        if value is not None:
            net_params[name] = typ(value)
    if task == 'zinc':
        if args.in_feat_dropout is not None:
            net_params['in_feat_dropout'] = float(args.in_feat_dropout)
        if args.edge_feat is not None:
            net_params['edge_feat'] = _BOOL(args.edge_feat)
    if net_params['pe_init'] != 'prism' and args.prism_preset is None:
        net_params.pop('prism_preset', None)
    if args.early_stopping_patience > 0:
        params['early_stopping_patience'] = args.early_stopping_patience
    if args.gpu_id is not None:
        config['gpu']['id'] = args.gpu_id
        config['gpu']['use'] = True
    if args.model is not None:
        config['model'] = args.model
    if args.dataset is not None:
        config['dataset'] = args.dataset
    if args.out_dir is not None:
        config['out_dir'] = args.out_dir
    if args.smoke:
        params['epochs'] = 1
        config['out_dir'] = os.path.join(config['out_dir'], 'smoke') + os.sep
    elif not config['out_dir'].endswith(os.sep):
        config['out_dir'] += os.sep
    seeds = [args.seed] if args.seed is not None else ([SEEDS[0]] if args.smoke else list(SEEDS))
    return config, params, net_params, seeds
