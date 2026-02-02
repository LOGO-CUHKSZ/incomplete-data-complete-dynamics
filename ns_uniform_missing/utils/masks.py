import torch
from torch import Tensor

__all__ = ['uniform_mask']

def uniform_mask(x: Tensor, p: float) -> Tensor:
    assert x.ndim == 4
    # x.shape == (batch_size, n_time, img_size, img_size)
    while True:
        mask = torch.rand((x.size(0), 1, *x.shape[2:]), device=x.device) < p
        if mask.reshape(mask.size(0), -1).sum(-1).min() > 4:
            return mask.repeat(1, x.size(1), 1, 1)