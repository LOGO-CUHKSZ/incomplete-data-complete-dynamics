import os
import gc
import sys
import time
import yaml
import shutil
import warnings

from easydict import EasyDict

import torch

from utils.runner import train
from utils.data_loader import ns_loader
from utils.utils import seed_all, set_cuda

from models.ema import EMA
from models.sde import *

from sampling import sampling

def main(conf_path: str, is_coding_mode: bool = False) -> str:
    with open(conf_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    seed_all(config.train.seed)
    log_dir = conf_path.split('/')[:-1]
    current_file = os.path.relpath(__file__)
    shutil.copy(current_file, os.path.join(*log_dir, current_file))
    shutil.copytree('./models', os.path.join(*log_dir, 'models'))
    shutil.copytree('./utils', os.path.join(*log_dir, 'utils'))
    ckpt_dir = os.path.join(*log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(config.train.device)
    train_loader, val_loader = ns_loader(
        config.dataset,
        batch_size=config.train.batch_size,
        num_workers=0,
    )

    match config.sde.sde:
        case 'dm':   generative_model = DM
        case _: raise NotImplementedError()

    
    model = generative_model(
        config.sde, config.model, config.learning, train_loader.dataset
    ).to(device)
    
    ema = EMA(model, decay=0.999) if config.train.ema else None

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0,
        betas=(0.95, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.6,
        patience=5//config.train.val_freq,
    )

    model_save_path = train(
        ckpt_dir, device, model, ema, config,
        train_loader, val_loader, optimizer, scheduler
    )

    del model, ema, optimizer, scheduler
    gc.collect()

    if config.learning.model == 'ldpv':
        sampling_configs = [
            {'steps': 1, 'sampling': {'random_context': True}},
            {'steps': 1, 'sampling': {'random_context': False}}
        ]
        for sampling_cfg in sampling_configs:
            gc.collect()
            sampling(model_save_path, sampling_config=sampling_cfg, device=device)
        gc.collect()
    return 

def get_arg(
        sde: dict,
        model: dict,
        learning: dict,
        train_config: dict,
        dataset: str,
        sampling: dict | None = None,
        is_coding_mode: bool = False,
        save_path: str = 'logs',
    ) -> tuple[str, bool]:

    assert model.__contains__('model') and model.__contains__('network')

    yml = {}
    yml['sde'] = sde
    yml['model'] = model
    yml['learning'] = learning
    yml['train'] = train_config
    yml['dataset'] = dataset
    yml['sampling'] = sampling
    
    now_time = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if is_coding_mode:
        save_folder = os.path.join(save_path, os.path.join('temp', now_time))
    else:
        key_settings = [
            dataset.split('/')[-1].rstrip('.h5'),
            learning['model'], model['model'], sde['sde']
        ]
        save_folder = key_settings + [sde['matching'], f"seed{train_config['seed']}"]
        save_folder = '---'.join(str(item) for item in save_folder)
    
        yml['project'] = save_folder
        save_folder = save_folder + '---' + now_time
        save_folder = os.path.join(save_path, *key_settings, save_folder)

    os.makedirs(save_folder, exist_ok=True)
    save_yml = os.path.join(save_folder, 'config.yml')
    with open(save_yml, 'w') as yaml_file:
        yaml.dump(yml, yaml_file, sort_keys=False)

    return save_yml, is_coding_mode



if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
    torch.multiprocessing.set_start_method('spawn') 

    is_coding_mode = sys.gettrace() is not None

    sde_config = {
        'sde': 'dm',           # ['dm', 'fm']
        't_eps': 1e-3,
        'beta_min': 0.1,
        'beta_max': 20.0,
        'matching': 'data',    # ['noise', 'data']
    }


    learning_config = {
        'model': 'ldpv',
        'context_rm': 1,
        'query_unknown_only': True,
    }
    sampling_config = {
        'steps': 1,
        'sampling': {
            'random_context': True
        }
    }

    import sys
    m = int(sys.argv[1])
    assert m in [1, 4, 6]

    device = int(sys.argv[2])
    assert 0 <= device <= 7

    # m = 1
    # device = 0


    device = set_cuda(device)
    train_config = {
        'device': device,
        'seed': 42,
        'ema': False,
        'skip_batch': False,
        'batch_size': 8,
        'num_epoches': 1000,
        'val_freq': 2,
        'reload_bd': 1.5,
        'norm_clip': 1.0,
        'max_update_norm': 5.0
    }

    main(*get_arg(
        sde=sde_config,
        model={'model': 'karras', 'network': {'hidden_size': 32}},
        learning=learning_config,
        train_config=train_config,
        dataset=f'data/ns_block_{m}.h5',
        sampling=sampling_config,
        is_coding_mode=is_coding_mode
    ))













