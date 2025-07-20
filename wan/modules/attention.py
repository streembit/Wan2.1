import warnings
import torch
import torch.nn.functional as F

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ImportError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ImportError:
    FLASH_ATTN_2_AVAILABLE = False


def flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
    """
    Flash attention with fallback to PyTorch standard attention.
    
    Expected input shapes:
    - q: [B, Lq, Nq, C] or flattened
    - k: [B, Lk, Nk, C]
    - v: [B, Lk, Nk, C]
    """
    # Since flash_attn is not available, always use the fallback
    warnings.warn("Flash attention not available. Using PyTorch standard attention (slower and uses more memory)")
    
    # Determine if inputs are already flattened
    if q_lens is not None and k_lens is not None:
        # Inputs are flattened, need to unflatten
        b = q_lens.shape[0]
        
        # Calculate actual sequence lengths
        total_q = q.shape[0]
        total_k = k.shape[0]
        lq = total_q // b
        lk = total_k // b
        
        # Get other dimensions
        nq = q.shape[1]  # number of heads
        nk = k.shape[1]
        c = q.shape[2]   # head dimension
        
        # Reshape to [B, L, N, C]
        q = q.view(b, lq, nq, c)
        k = k.view(b, lk, nk, c)
        v = v.view(b, lk, nk, c)
    else:
        # Assume inputs are [B, L, N, C]
        b, lq, nq, c = q.shape
        b, lk, nk, c = k.shape
    
    # Apply dtype conversion
    orig_dtype = q.dtype
    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)
    
    # Apply query scale if provided
    if q_scale is not None:
        q = q * q_scale
    
    # For standard attention, we need [B, N, L, C] format
    q = q.transpose(1, 2)  # [B, Nq, Lq, C]
    k = k.transpose(1, 2)  # [B, Nk, Lk, C]
    v = v.transpose(1, 2)  # [B, Nk, Lk, C]
    
    # Handle multi-query/grouped-query attention if Nq != Nk
    if nq != nk:
        # Repeat k and v to match query heads
        assert nq % nk == 0, f"Number of query heads ({nq}) must be divisible by key heads ({nk})"
        repeat_factor = nq // nk
        k = k.repeat_interleave(repeat_factor, dim=1)  # [B, Nq, Lk, C]
        v = v.repeat_interleave(repeat_factor, dim=1)  # [B, Nq, Lk, C]
    
    # Create attention mask if using variable lengths
    attn_mask = None
    if q_lens is not None and k_lens is not None:
        # Create a mask for each sequence in the batch
        attn_mask = torch.zeros((b, lq, lk), dtype=torch.bool, device=q.device)
        for i in range(b):
            q_len = q_lens[i].item() if i < len(q_lens) else lq
            k_len = k_lens[i].item() if i < len(k_lens) else lk
            attn_mask[i, :q_len, :k_len] = True
        
        # Expand mask for all heads: [B, 1, Lq, Lk]
        attn_mask = attn_mask.unsqueeze(1)
        
        # Convert boolean mask to float mask (True -> 0, False -> -inf)
        attn_mask = torch.where(attn_mask, 0., float('-inf'))
    
    # Apply scaled dot product attention
    # Note: F.scaled_dot_product_attention expects [B, N, L, C] format
    output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        scale=softmax_scale,
        is_causal=causal and attn_mask is None  # Only use is_causal if no custom mask
    )
    
    # Transpose back to [B, L, N, C]
    output = output.transpose(1, 2)
    
    # If we need to return flattened output
    if q_lens is not None and k_lens is not None:
        output = output.reshape(total_q, nq, c)
    
    # Convert back to original dtype
    output = output.to(orig_dtype)
    
    return output


def _get_default_sliding_window_size():
    """Get default sliding window size"""
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return (-1, -1)
    else:
        # Use a reasonable window size for standard attention to save memory
        return (2048, 2048)
