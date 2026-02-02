import os
import h5py

import numpy as np

from numpy import ndarray
from typing import Literal
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader
from utils.masks import *

__all__ = [
    'NSDataset',
    'ns_loader'
]


class NSDataset(Dataset):
    def __init__(
            self,
            dpath: str, 
            mode: Literal['train', 'valid'],
            return_gt: bool,
            seed: int = 42
        ) -> None:
        super().__init__()
        self.dpath = dpath
        self.return_gt = return_gt

        with h5py.File(self.dpath, 'r') as h5f:
            x0 = h5f['samples'][:].astype(np.float32)
            masks = h5f['masks'][:]
            if return_gt:
                gts = h5f['gts'][:]

        rng = np.random.default_rng(seed=seed)
        arr = np.arange(len(x0))
        rng.shuffle(arr)
        match mode:
            case 'train':
                indices = arr[:int(0.9*len(arr))]
            case 'valid':
                indices = arr[int(0.9*len(arr)):]
            case _:
                raise ValueError()
            
        self.x0 = x0[indices]
        self.masks = masks[indices]
        if return_gt:
            self.gts = gts[indices]
        self.n_samples = len(self.x0)
        self.u_shape = self.x0.shape[1:]
            

    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx: int):
        if self.return_gt:
            return self.x0[idx], self.masks[idx], self.gts[idx]
        return self.x0[idx], self.masks[idx]

    
def ns_loader(
        dataset: str,
        num_workers: int = 4, batch_size: int = 256,
        return_gt: bool = False
    ) -> tuple[DataLoader, DataLoader]:

    train_dataset = NSDataset(dataset, 'train', return_gt)
    valid_dataset = NSDataset(dataset, 'valid', return_gt)

    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'multiprocessing_context': 'spawn' if num_workers else None,
        'generator': torch.Generator(device='cpu'),
        'persistent_workers': True if num_workers > 0 else False
    }

    return \
        DataLoader(train_dataset, **dataloader_kwargs), \
        DataLoader(valid_dataset, **dataloader_kwargs)
        
    

