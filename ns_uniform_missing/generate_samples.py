import os
import h5py

import torch
import numpy as np

from numpy import ndarray
from torch import Tensor

from tqdm import tqdm
from datetime import datetime
from utils.utils import seed_all
from utils.masks import *

from functools import partial

def generate_samples(
        data_path: str, 
        mask_cfg: dict, save_name: str,
        seed: int = 42
    ) -> None:
    seed_all(seed)
    mask, p = mask_cfg['mask'], mask_cfg['p']
    assert not save_name.endswith('.h5')
    save_name = save_name + f'_{mask}_{int(100*p)}.h5'

    data = torch.load(data_path, map_location='cpu')['vorticity'] / 38.0
    data = data.unsqueeze(1)

    match mask:
        case 'uniform':
            mask_func = partial(uniform_mask, p=mask_cfg['p'])
        case _:
            raise ValueError()

    n_samples = len(data)
    with h5py.File(save_name, 'w') as f:
        f.attrs['n_samples'] = n_samples
        f.attrs['creation_date'] = datetime.now().isoformat()

        samples = f.create_dataset('samples', shape=data.shape, dtype='float32')
        masks = f.create_dataset('masks', shape=data.shape, dtype='bool')
        gts = f.create_dataset('gts', shape=data.shape, dtype='float32')

        mask = mask_func(data)
        gts[:] = data.numpy()
        samples[:] = (mask * data).numpy()
        masks[:] = mask.numpy()

    print('Done!!')

def uniform_mask(x: Tensor, p: float) -> Tensor:
    assert x.ndim == 5
    # x.shape == (batch_size, n_channels, n_time, img_size, img_size)
    while True:
        mask = torch.rand((x.size(0), 1, 1, *x.shape[3:]), device=x.device) < p
        if mask.reshape(mask.size(0), -1).sum(-1).min() > 4:
            return mask.repeat(1, 1, x.size(2), 1, 1)

if __name__ == "__main__":
    mask = 'uniform'
    for p in tqdm([0.2, 0.6, 0.8]):
        mask_cfg = {'mask': mask, 'p': p}
        generate_samples(
            'data/Kolmogorov2d_fp32_64x64_N1152_Re1000_T100.pt', mask_cfg, 'data/ns'
        )

