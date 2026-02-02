import h5py

import torch

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
    mask, m = mask_cfg['mask'], mask_cfg['m']
    assert not save_name.endswith('.h5')
    save_name = save_name + f'_{mask}_{m}.h5'

    data = torch.load(data_path, map_location='cpu')['vorticity'] / 38.0
    data = data.unsqueeze(1)

    match mask:
        case 'block':
            mask_func = partial(block_mask, m=mask_cfg['m'])
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




if __name__ == "__main__":
    mask = 'block'
    for m in tqdm([1, 4, 6]):
        mask_cfg = {'mask': mask, 'm': m}
        generate_samples(
            'data/Kolmogorov2d_fp32_64x64_N1152_Re1000_T100.pt', mask_cfg, 'data/ns'
        )

