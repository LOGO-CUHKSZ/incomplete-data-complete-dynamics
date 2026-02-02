import os
import pickle

import numpy as np

from tqdm import tqdm

from models.inpainting import video_inpainting

from utils.data_loader import ns_loader

from numpy import ndarray
from torch import Tensor
from typing import Callable

def inverse_transform_11(x: ndarray, low: ndarray, high: ndarray) -> ndarray:
    return 0.5 * ((high - low) * x + low + high)

def transform_01(x: ndarray, low: ndarray, high: ndarray) -> ndarray:
    return (x - low) / (high - low)

def pkl_save(data, save_name: str) -> None:
    with open(save_name, 'wb') as f:
        pickle.dump(data, f)

def inpaint(
        dataset: str,
        save_folder: str,
        method: str
    ) -> None:
    test_loader = ns_loader(
        dataset,
        # batch_size=4*config.train.batch_size,
        batch_size=32,
        num_workers=0,
        return_gt=True
    )[-1]
    

    torch2np: Callable[[Tensor], ndarray] = lambda x: x.detach().cpu().numpy()

    output_dir = os.path.join(save_folder, dataset, method)
    os.makedirs(output_dir, exist_ok=True)

    results, gts, masks = [], [], []
    pbar = tqdm(test_loader, dynamic_ncols=True)
    for i, batch in enumerate(pbar):
        pbar.set_description(f'Solve using method: {method}')
        x0, mask, gt = batch
        mask = np.repeat(mask, x0.shape[1], 1)
        assert x0.shape == mask.shape
        # x0 in the range of [-1, 1]
        results.append(
            2 * video_inpainting(
                0.5 * (torch2np(x0) + 1),
                torch2np(mask),
                method=method
            ) - 1
        )
        gts.append(torch2np(gt))
        masks.append(torch2np(mask))

    results, gts, masks = map(np.concatenate, [results, gts, masks])

    pkl_save(gts, os.path.join(output_dir, 'gts_all.pkl'))
    pkl_save(masks, os.path.join(output_dir, 'masks_all.pkl'))

    error = float(np.square((results - gts)).mean())
    samples_path = os.path.join(output_dir, f'samples_all_{error}.pkl')
    pkl_save(np.concatenate(results), samples_path)


if __name__ == '__main__':
    # import sys
    # try:
    #     dataset = int(sys.argv[1])
    #     dataset = f'data/ns_block_{dataset}.h5'
    # except:
    #     dataset = sys.argv[1]
    
    save_folder = 'classical_inpainting'
    dataset = 'data/ns_block_4.h5'
    methods = [
        'navier_stokes',
        'fast_marching',
        'temporal_consistency',
    ]
    for method in methods:
        print(dataset, save_folder, method)
        inpaint(dataset, save_folder, method)
