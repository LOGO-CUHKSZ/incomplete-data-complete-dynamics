import torch
from torch import Tensor
from typing import List, Optional, Union

__all__ = ['block_mask', 'infer_block_ids_from_mask']

def block_mask(
    x: Tensor,
    m: int,
    block_ids: Optional[Union[List[int], Tensor]] = None
) -> Tensor:
    """
    Generate a mask by masking blocks in a 3x3 partitioned image.

    Args:
        x: Tensor of shape (B, C, T, H, W)
        m: Number of blocks to mask (used if block_ids is None)
        block_ids: Optional. Can be:
            - None: randomly mask m blocks per sample
            - List[int]: mask same block_ids for all samples
            - Tensor of shape (B, k): each row specifies block_ids for each sample

    Returns:
        mask: Bool tensor of shape (B, 1, T, H, W), where False means masked
    """
    assert x.ndim == 5
    B, C, T, H, W = x.shape

    h_splits = [H // 3] * 3
    w_splits = [W // 3] * 3
    h_splits[-1] += H - sum(h_splits)
    w_splits[-1] += W - sum(w_splits)
    h_start = [sum(h_splits[:i]) for i in range(3)]
    w_start = [sum(w_splits[:j]) for j in range(3)]

    mask = torch.ones((B, 1, 1, H, W), dtype=torch.bool, device=x.device)

    for b in range(B):
        if block_ids is None:
            selected = torch.randperm(9, device=x.device)[:m].tolist()
        elif isinstance(block_ids, list):
            selected = block_ids
        elif isinstance(block_ids, Tensor):
            selected = block_ids[b].tolist()
        else:
            raise TypeError("block_ids must be None, List[int], or Tensor of shape (B, k)")

        for idx in selected:
            assert 0 <= idx <= 8, f"Block index {idx} out of range"
            i, j = divmod(idx, 3)
            hs, he = h_start[i], h_start[i] + h_splits[i]
            ws, we = w_start[j], w_start[j] + w_splits[j]
            mask[b, 0, 0, hs:he, ws:we] = False

    return mask.expand(B, 1, T, H, W)


def infer_block_ids_from_mask(mask: Tensor, threshold: float = 0.5) -> List[List[int]]:
    """
    Infer which 3x3 blocks (index 0-8) are masked for each sample based on a threshold.

    Args:
        mask: Bool tensor of shape (B, 1, T, H, W), where False = masked
        threshold: Proportion of pixels that must be masked (False) to consider a block masked

    Returns:
        List of length B, each is a list of masked block indices (0-8)
    """
    B, _, T, H, W = mask.shape
    assert H >= 3 and W >= 3, "Image must be large enough for 3x3 partitioning"

    # Compute split sizes and start positions (must match block_mask logic)
    h_splits = [H // 3] * 3
    w_splits = [W // 3] * 3
    h_splits[-1] += H - sum(h_splits)
    w_splits[-1] += W - sum(w_splits)
    h_start = [sum(h_splits[:i]) for i in range(3)]
    w_start = [sum(w_splits[:j]) for j in range(3)]

    block_ids_list = []

    for b in range(B):
        masked_blocks = []
        for idx in range(9):
            i, j = divmod(idx, 3)
            hs, he = h_start[i], h_start[i] + h_splits[i]
            ws, we = w_start[j], w_start[j] + w_splits[j]

            block = mask[b, 0, :, hs:he, ws:we]  # shape (T, h, w)
            total = block.numel()
            masked = (~block).sum().item()

            if masked / total >= threshold:
                masked_blocks.append(idx)

        block_ids_list.append(masked_blocks)

    return block_ids_list

