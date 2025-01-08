from typing import Optional

import mlx.core as mx
import numpy as np


def create_4d_attention_mask_for_sdpa(
    mask: np.ndarray, dtype: np.dtype, target_len: Optional[int] = None
):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Inpsired by: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_attn_mask_utils.py#L429

    Args:
        mask (`no.ndarray`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`np.dtype`):
            The mlx dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    batch_size, seq_len = mask.shape
    target_len = target_len if target_len is not None else seq_len

    mask_exp = mx.broadcast_to(
        mask[:, None, :], shape=(batch_size, 1, target_len, seq_len)
    ).astype(dtype)
    mask_inv = 1.0 - mask_exp
    # TODO: use mx.finfo once >0.21.1 drops
    mask_inv = mx.where(mask_inv == 0.0, mask_inv, float(np.finfo(np.float32).min))
    return mask_inv
