import torch

from torch import Tensor


__all__ = [
    'diffusion_fields'
]

def diffusion_fields(
    xt: Tensor,
    xt_mask: Tensor,
    context_mask: Tensor | None = None, 
    query_mask: Tensor | None = None,
    context_ratio: float | None = None, 
    query_ratio: float | None = None,
    **kwargs
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    assert xt.ndim == 5
    mask_given = (context_mask is not None) and (query_mask is not None)
    ratio_given = (context_ratio is not None) and (query_ratio is not None)

    assert mask_given != ratio_given, (
        "You must provide either both context_mask and query_mask, "
        "or both context_ratio and query_ratio, but not both or neither."
    )
    if ratio_given:
        assert 0 < context_ratio < 1 and query_ratio > 0
        bool_xt_mask = xt_mask.bool()
        context_mask = bool_xt_mask & (torch.rand(xt_mask.shape, device=xt_mask.device) < context_ratio)
        query_mask   = bool_xt_mask & (torch.rand(xt_mask.shape, device=xt_mask.device) < query_ratio)
    else:
        assert not (context_mask & ~xt_mask).any()
        assert not (query_mask   & ~xt_mask).any()
    context = context_mask * xt
    query   = query_mask * xt
    return context, query, context_mask, query_mask

















