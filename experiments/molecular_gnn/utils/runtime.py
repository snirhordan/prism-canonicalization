"""Device setup, seeding and small helpers shared by the two entry points."""

import os
import random
import time

import numpy as np
import torch


def gpu_setup(use_gpu, gpu_id):
    """Select the device.

    ``CUDA_VISIBLE_DEVICES`` set by the scheduler (SLURM) is respected; otherwise it is set to
    ``gpu_id``.  In both cases the process then sees its GPU as ``cuda:0``.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model):
    return int(sum(int(np.prod(list(p.data.size()))) for p in model.parameters()))


def run_paths(out_dir, model_name, dataset_name, gpu_id, seed):
    """Result and config file stems of one run (seed included, so seeds never overwrite each other)."""
    stamp = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    name = '%s_%s_GPU%s_seed%d_%s' % (model_name, dataset_name, gpu_id, seed, stamp)
    os.makedirs(os.path.join(out_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'configs'), exist_ok=True)
    return (os.path.join(out_dir, 'results', 'result_' + name),
            os.path.join(out_dir, 'configs', 'config_' + name))
