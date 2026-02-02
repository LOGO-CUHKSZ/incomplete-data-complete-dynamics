import os
import wandb

import numpy as np

from tqdm import tqdm
from easydict import EasyDict
from collections import deque
from typing import Any

import torch

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from models.ema import EMA
from models.sde import DM
from utils.utils import move_to_device

__all__ = ['train']

def train(
        ckpt_dir: str,
        device: torch.device | int, 
        model: DM, ema: EMA | None,
        config: EasyDict,
        train_loader: DataLoader, val_loader: DataLoader,
        optimizer: Optimizer, scheduler: LRScheduler
    ) -> str:
    model = model.to(device)
    best_val_loss = float('inf')
    print(' Training start!!!')

    train_que_len, val_que_len = 40, 5
    train_grad_queue = deque([np.inf] * train_que_len, maxlen=train_que_len)
    train_loss_queue = deque([np.inf] * train_que_len, maxlen=train_que_len)
    val_loss_queue = deque([np.inf] * val_que_len, maxlen=val_que_len)

    ckpt_path = os.path.join(ckpt_dir, f'ckpt.pt')

    train_kwargs = {
        'skip_batch': config.train.skip_batch,
        'optimizer': optimizer,
        'grad_queue': train_grad_queue,
        'loss_queue': train_loss_queue,
        'epoch': 0, 'device': device,
        'large_grad_counts': 0,
        'reload_path': ckpt_path,
        'norm_clip': config.train.norm_clip,
        'max_update_norm': config.train.max_update_norm
    }
    
    for epoch in tqdm(range(config.train.num_epoches), dynamic_ncols=True):
        train_kwargs['epoch'] = epoch
        train_kwargs = train_epoch_(model, ema, train_loader, **train_kwargs)

        if epoch % config.train.val_freq == 0:
            val_loss = val_epoch(model, ema, val_loader, epoch, device)
            scheduler.step(val_loss)
            val_loss_queue.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'ema': ema.shadow if ema else None,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'avg_val_loss': val_loss,
                    'train_grad_queue': train_grad_queue,
                    'train_loss_queue': train_loss_queue,
                    'val_loss_queue': val_loss_queue
                }, ckpt_path)

            min_reload_loss = min(
                3 * np.mean(val_loss_queue),
                5 * best_val_loss,
                config.train.reload_bd
            )

            if (val_loss > min_reload_loss or np.isnan(val_loss) or np.isinf(val_loss)) and \
                os.path.exists(ckpt_path):
                print('\n Training failed (val). Loading model histroy!')
                ckpt_data = torch.load(ckpt_path)
                model.load_state_dict(ckpt_data['model'], strict=False)
                if ckpt_data['ema'] is not None:
                    ema.shadow = ckpt_data['ema']
                optimizer.load_state_dict(ckpt_data['optimizer'])

                val_loss = ckpt_data['avg_val_loss']
                train_grad_queue = ckpt_data['train_grad_queue']
                train_loss_queue = ckpt_data['train_loss_queue']
                val_loss_queue   = ckpt_data['val_loss_queue']

        if optimizer.param_groups[0]['lr'] < 5e-7:
            print(' Model converges! Early stopping!')
            break


    return ckpt_path


def train_epoch_(
        model: DM, ema: EMA | None,
        train_loader: DataLoader, optimizer: Optimizer,
        grad_queue: deque, loss_queue: deque,
        epoch: int, device: torch.device,
        skip_batch: bool, norm_clip: float, max_update_norm: float,
        large_grad_counts: int = 0,
        reload_path: str | None = None
    ) -> dict[str, Any]:
    model.train()
    pbar = tqdm(train_loader, leave=False, dynamic_ncols=True)
    lr = float('inf')

    for i, batch in enumerate(pbar):
        loss_dict = model.get_loss(move_to_device(batch, device))[0]
        loss = loss_dict['total']
        loss.backward()

        lr, train_loss = optimizer.param_groups[0]['lr'], loss.item()
        pbar.set_description(f'Training loss: {"{:.6e}".format(train_loss)} lr: {"{:.4e}".format(lr)}')

        mean_grad = np.mean(grad_queue)
        orig_grad_norm = clip_grad_norm_(model.parameters(), min(norm_clip, 3 * mean_grad)).item()

        if skip_batch:
            if epoch < 2 or (
                train_loss < 5 * np.mean(loss_queue) \
                    and \
                not np.isnan(orig_grad_norm) \
                    and \
                orig_grad_norm < max_update_norm
                ):
                large_grad_counts = 0

                optimizer.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update()

                grad_queue.append(orig_grad_norm)
                loss_queue.append(train_loss)
            else:
                large_grad_counts += 1
        else:
            optimizer.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update()

        if (not skip_batch) and large_grad_counts > 5 and os.path.exists(reload_path):
            large_grad_counts = 0
            print('\n Training failed (train). Loading model histroy!')
            ckpt_data = torch.load(reload_path)
            model.load_state_dict(ckpt_data['model'], strict=False)
            grad_queue = ckpt_data['train_grad_queue']
            loss_queue = ckpt_data['train_loss_queue']

        if wandb.run is not None:
            wandb.log({
                'train/lr': lr,
                'train/grad_norm': orig_grad_norm,
                'custom_step': epoch,
            } | {f'train/{k}': v.item() for k, v in loss_dict.items()})

    return {
        'skip_batch': skip_batch,
        'optimizer': optimizer,
        'grad_queue': grad_queue,
        'loss_queue': loss_queue,
        'epoch': epoch, 'device': device,
        'large_grad_counts': large_grad_counts,
        'reload_path': reload_path,
        'norm_clip': norm_clip,
        'max_update_norm': max_update_norm
    }

def val_epoch(
        model: DM, ema: EMA | None,
        val_loader: DataLoader, epoch: int, device: torch.device
    ) -> float:
    sum_loss, sum_n = 0, 0
    if ema is not None:
        ema.apply_ema_weights()
    model.eval()

    pbar = tqdm(val_loader, desc='Validation', leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            loss_dict, batch_size = model.get_loss(move_to_device(batch, device))

            if loss_dict.__contains__('score'):
                loss = loss_dict['score']
            else:
                loss = loss_dict['total']
            sum_loss += loss.item() * batch_size
            sum_n += batch_size
    avg_loss = sum_loss / sum_n

    if wandb.run is not None:
        wandb.log({
            'val/loss': avg_loss,
            'custom_step': epoch
        })
    if ema is not None:
        ema.restore_original_weights()
    return avg_loss

