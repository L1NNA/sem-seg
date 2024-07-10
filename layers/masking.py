import torch
from typing import Optional


def create_masking(seq_len:int, span_len:int, device:Optional[torch.device]=None,
                   mem_len:Optional[int]=None) -> torch.Tensor:
    """Generate a look-ahead (or look-back) mask
    to prevent attention to future tokens
    (or old tokens if necessary).

    Args:
        seq_len: input sequence length
        span_len: the length of the current span including input and memory
        device: tensor device
        mem_len: if provided, the current timestep can only look back
        the same memory length

    Returns:
        the look ahead mask
    """
    assert seq_len <= span_len
    offset = span_len - seq_len
    mask = torch.ones(seq_len, span_len, device=device).triu(offset + 1)
    if mem_len is not None:
        mask_len = span_len - mem_len
        mask_shift_len = (seq_len - mask_len) if mask_len > 0 else seq_len
        mask += torch.ones(seq_len, mem_len, device=device).tril(-mask_shift_len)
    return mask.bool()


def create_segment_masking(seg_ids:torch.Tensor) -> torch.Tensor:
    """

    Args:
        seg_ids: b x w

    Returns:
        b x h x w x w
    """
    b, w = seg_ids.size()
    seg_ids[:, -1] = 1 # force the last token to be a boundary

    masking = torch.ones(b, w, w, device=seg_ids.device)

    indices = torch.where(seg_ids == 1)
    coordinates = list(zip(indices[0].cpu().numpy(), indices[1].cpu().numpy()))

    last_i, last_j = 0, 0
    for i, j in coordinates:
        if i != last_i:
            last_i = i
            last_j = 0
        masking[i, last_j:j+1, last_j:j+1] = 0
        last_j = j+1
    return masking.bool().unsqueeze(1)
    
