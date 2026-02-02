import math
import torch
import torch.nn as nn

from torch import Tensor

from models.FNO.models import FNO3d, TFNO3d

from functools import partialmethod

__all__ = [
    'FNO3D'
]

class FNO3D(nn.Module):
    def __init__(
            self,
            image_size: int,
            input_channels: int,
            output_channels: int,
            **kwargs
        ):
        super().__init__()
        t_emb_dim = 128

        self.fno = FNO3d(
            n_modes_height=input_channels,
            n_modes_width=image_size,
            n_modes_depth=image_size,
            in_channels=input_channels+t_emb_dim,
            out_channels=output_channels,
            **kwargs
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.GELU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )


    def forward(
            self,
            t: Tensor,
            xt: Tensor,
            mask: Tensor
        ):
        # t.shape == (batch_size, )
        # xt.shape == (batch_size, n_channel, n_time, img_size, img_size)

        cond = self.time_mlp(t)
        cond = cond.view(*cond.shape, *[1]*(xt.ndim-2)).repeat(1, 1, xt.size(2), *xt.shape[3:])

        model_input = torch.cat((
            mask.float(), xt,
            cond
        ), 1)
        return self.fno(model_input)



class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

def partialclass(new_name, cls, *args, **kwargs):
    """Create a new class with different default values

    Notes
    -----
    An obvious alternative would be to use functools.partial
    >>> new_class = partial(cls, **kwargs)

    The issue is twofold:
    1. the class doesn't have a name, so one would have to set it explicitly:
    >>> new_class.__name__ = new_name

    2. the new class will be a functools object and one cannot inherit from it.

    Instead, here, we define dynamically a new class, inheriting from the existing one.
    """
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    new_class = type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )
    return new_class