import os
import yaml
import time
import pickle
import numpy as np

import torch

from glob import glob
from tqdm import tqdm

from numpy import ndarray
from torch import Tensor
from typing import Callable
from easydict import EasyDict

from utils.data_loader import ns_loader
from utils.utils import seed_all, set_cuda
from utils.masks import *
from models.sde import *

from functools import partial

__all__ = [
    'sampling'
]

def pkl_save(data, save_name: str) -> None:
    with open(save_name, 'wb') as f:
        pickle.dump(data, f)

def sampling(
        ckpt_path: str,
        sampling_config: dict | None = None,
        device: int | None = None
    ) -> str:
    device = torch.device(set_cuda(device))
    ckpt = torch.load(ckpt_path, map_location=device)

    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    seed_all(config.train.seed)

    if sampling_config is None:
        sampling_config = config['sampling']

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(ckpt_path)),
        'generated_samples-' + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    )
    os.makedirs(output_dir)

    test_loader = ns_loader(
        config.dataset,
        batch_size=4*config.train.batch_size,
        # batch_size=2,
        num_workers=0,
        return_gt=True
    )[-1]
    
    match config.sde.sde:
        case 'dm':   generative_model = DM
        case _: raise ValueError()
    
    model = generative_model(
        config.sde, config.model, config.learning, test_loader.dataset
    ).to(device)

    model.load_state_dict(ckpt['model'], strict=False)
    if ckpt.__contains__('ema') and ckpt['ema'] is not None:
        for name, param in model.named_parameters():
            param.data.copy_(ckpt['ema'][name])
            
    torch2np: Callable[[Tensor], ndarray] = lambda x: x.detach().cpu().numpy()

    results, gts, masks = [], [], []
    for batch in tqdm(test_loader, dynamic_ncols=True):
        x0, mask, gt = batch

        results.append(torch2np(
            model.sampling(
                x0.to(device),
                mask.to(device),
                **sampling_config
            )
        ))
        gts.append(torch2np(gt))
        masks.append(torch2np(mask))



    results, gts, masks = map(np.concatenate, [results, gts, masks])
    masks = np.repeat(masks, gts.shape[1], 1)

    pkl_save(gts, os.path.join(output_dir, 'gts_all.pkl'))
    pkl_save(masks, os.path.join(output_dir, 'masks_all.pkl'))

    # error = float(np.round(
    #     np.square((results - gts)[~masks]).mean(),
    #     decimals=5
    # ))
    error = float(np.square((results - gts)).mean())

    samples_path = os.path.join(output_dir, f'samples_all.pkl')
    pkl_save(results, samples_path)
    with open(os.path.join(output_dir, 'sampling_config.yml'), 'w') as yaml_file:
        yaml.dump(sampling_config | {'error': error}, yaml_file, sort_keys=False)

    return samples_path
        

if __name__ == '__main__':
    import gc
    # sampling_config = {'steps': 1000}

    # sampling_config = {
    #     'steps': 200,
    #     'sampling': {
    #         'random_context': True
    #     }
    # }
    # sampling_config = {
    #     'steps': 1,
    #     'sampling': {
    #         'random_context': False
    #     }
    # }
    sampling_config = {
        'steps': 200,
        'sampling': {
            'weighted': 'linear',
            'random_context': True
        }
    }



    device = 1

    # root = 'logs'
    # for dataset in os.listdir(root):
    #     if dataset == 'temp':
    #         continue
    #     m = int(dataset.split('_')[-1].rstrip('.h5'))
    #     sampling_config['mask_func'] = partial(block_mask, m=m)
    #     for learning in os.listdir(os.path.join(root, dataset)):
    #         if learning == 'dsm':
    #             continue
    #         for model in os.listdir(os.path.join(root, dataset, learning)):
    #             if model == 'fno': 
    #                 continue
                
    #             for sde in os.listdir(os.path.join(root, dataset, learning, model)):
    #                 for experiment in os.listdir(os.path.join(root, dataset, learning, model, sde)):
    #                     if experiment.__contains__('noise'):
    #                         continue
    #                     ckpt_path = os.path.join(
    #                         root, dataset, learning, model, sde, experiment,
    #                         'checkpoints', 'ckpt.pt'
    #                     )

    #                     print(ckpt_path)
    #                     gc.collect()
    #                     sampling(ckpt_path, sampling_config, device)




    model_path = 'logs/ns_block_1/ldpv/karras/dm/ns_block_1---ldpv---karras---dm---data---seed42---2025_07_09__03_24_21/checkpoints/ckpt.pt'

    # sampling_config = {
    #     'steps': 1000,
    #     'sampling': {
    #         'random_context': True
    #     }
    # }

    sampling_config = {
        'steps': 200,
        'sampling': {
            # 'weighted': 'linear',
            'random_context': True
        }
    }
    sampling_config['mask_func'] = partial(block_mask, m=1)

    gc.collect()
    sampling(model_path, sampling_config)
