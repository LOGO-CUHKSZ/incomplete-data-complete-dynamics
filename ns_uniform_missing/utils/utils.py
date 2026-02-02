import os
import yaml
import random
import pynvml
import numpy as np

from glob import glob
from easydict import EasyDict
from itertools import product
from typing import Sequence

import torch

__all__ = [
    'load_config',
    'set_cuda',
    'seed_all',
    'grid_search',
    'move_to_device'
]


def load_config(ckpt_path: str) -> EasyDict:
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        return EasyDict(yaml.safe_load(f))

def set_cuda(
        device: int|str|torch.device|Sequence[int|str]|None=None
    ) -> str:
    if isinstance(device, int | str | torch.device | None):
        return _choose_cuda(device)
    else:
        raise ValueError()


def _choose_cuda(device: int | str | torch.device | None = None) -> str:
    assert torch.cuda.is_available()
    pynvml.nvmlInit()
    to_gb = 1024**3
    if device is None:
        free = []
        n_gpu = pynvml.nvmlDeviceGetCount()
        for gpu_id in range(n_gpu):
            handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
            free.append(meminfo.free / to_gb)
        device = np.argsort(free)[-1]
    elif isinstance(device, torch.device):
        device = int(str(device).lstrip('cuda:'))
    elif isinstance(device, int):
        assert device >= 0
    elif isinstance(device, str):
        device = int(device.lstrip('cuda:'))
    else:
        raise ValueError
    device: int
        
    handler = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    print(f'Using device: {device}. Free memory: {meminfo.free / to_gb}')
    return 'cuda:' + str(device)

def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def grid_search(param_grid: dict) -> list[dict]:
    keys, values = param_grid.keys(), param_grid.values()
    return [dict(zip(keys, combination)) for combination in product(*values)]


def move_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(item, device) for item in batch)
    else:
        return batch
    

