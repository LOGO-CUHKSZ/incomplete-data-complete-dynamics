import torch
import random

from torch import Tensor
from utils.masks import *

__all__ = [
    'diffusion_fields'
]

def add_random_blocks(original: Tensor, n: int) -> tuple[Tensor, Tensor]:
    """
    Add `n` new non-repeating random integers from [0, 8] to each row of `original`.

    Args:
        original: (B, m) tensor of unique integers in 0-8
        n: number of new elements to add per row

    Returns:
        (B, m+n) tensor with expanded unique entries per row
    """
    B, m = original.shape
    assert m + n <= 9, "Cannot have more than 9 unique values from [0, 8]"

    expanded, new_ids = [], []
    for i in range(B):
        current_set = set(original[i].tolist())
        available = list(set(range(9)) - current_set)
        new_blocks = random.sample(available, n)
        expanded_row = original[i].tolist() + new_blocks
        expanded.append(expanded_row)
        new_ids.append(new_blocks)

    return torch.tensor(expanded, dtype=torch.long, device=original.device), torch.tensor(new_ids, dtype=torch.long, device=original.device)


def diffusion_fields(
    xt: Tensor,
    xt_mask: Tensor,
    context_rm: float | None = None,
    query_unknown_only: bool = False,
    **kwargs
) -> tuple[Tensor, Tensor, Tensor]:
    assert xt.ndim == 5

    mask_ids = torch.tensor(infer_block_ids_from_mask(xt_mask), dtype=torch.int, device=xt.device)
    assert xt.size(0) == mask_ids.size(0)

    ctx_ids, new_ids = add_random_blocks(mask_ids, context_rm)
    ctx_mask = block_mask(xt, None, ctx_ids)
    qry_mask = ~block_mask(xt, None, new_ids) if query_unknown_only else xt_mask

    return ctx_mask * xt, ctx_mask, qry_mask

















