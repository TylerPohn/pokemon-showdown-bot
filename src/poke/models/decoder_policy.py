"""Decoder-only causal transformer policy for Pokemon battles.

This implements a Metamon-style architecture with:
- Decoder-only causal transformer (~200M parameters)
- RoPE (Rotary Position Embeddings)
- SwiGLU activation in FFN
- Pre-LayerNorm architecture
- HL-Gauss value classification head
- Flash Attention support (when available)
"""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ScaledEncoderConfig


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for relative position encoding.

    RoPE encodes position information directly into the attention mechanism
    through rotation of query and key vectors, enabling better extrapolation
    to longer sequences than absolute position embeddings.
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Interleave: [cos, sin, cos, sin, ...]
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for sequence positions.

        Args:
            x: Input tensor (for device/dtype)
            seq_len: Current sequence length

        Returns:
            Tuple of (cos, sin) tensors for rotation
        """
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, :].to(x.dtype),
            self.sin_cached[:, :, :seq_len, :].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV caching.

    Features:
    - Rotary position embeddings
    - Causal masking (can only attend to past)
    - KV cache for efficient autoregressive inference
    - Optional Flash Attention when available
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection (combined for efficiency)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

        # RoPE
        self.rotary = RotaryPositionalEmbedding(self.head_dim, max_seq_len)

        # Check for Flash Attention availability
        self.use_flash = use_flash_attention and hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with optional KV caching.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            kv_cache: Previous KV cache for incremental decoding
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of (output, new_kv_cache)
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Get rotary embeddings
        cos, sin = self.rotary(q, seq_len)

        # Handle KV cache for incremental decoding
        if kv_cache is not None:
            # Only apply RoPE to new positions
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # Concatenate with cached KV
            k = torch.cat([kv_cache["k"], k], dim=2)
            v = torch.cat([kv_cache["v"], v], dim=2)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Build new cache if needed
        new_cache = None
        if use_cache:
            new_cache = {"k": k, "v": v}

        # Attention computation
        if self.use_flash and not use_cache:
            # Use Flash Attention (PyTorch 2.0+)
            # Note: Flash attention handles causal masking internally
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Standard attention with explicit causal mask
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Causal mask
            kv_len = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)

        return output, new_cache


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    SwiGLU (Swish-Gated Linear Unit) is more efficient than GELU
    and provides better performance for transformers.

    Architecture: x -> [SiLU(W1 * x) * (W2 * x)] -> W3 -> output
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # SwiGLU uses 2/3 of the FFN dimension for the gate
        # This maintains parameter count similar to standard FFN
        hidden_dim = int(2 * d_ff / 3)
        # Round to nearest multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)  # Up projection
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)  # Down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class DecoderBlock(nn.Module):
    """Single decoder block with pre-LayerNorm architecture.

    Architecture:
        x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
             |________________________|    |___________________|
                     (residual)                  (residual)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, dropout, max_seq_len, use_flash_attention
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # Attention with residual
        normed = self.norm1(x)
        attn_out, new_cache = self.attn(normed, kv_cache, use_cache)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        return x, new_cache


class DecoderPolicy(nn.Module):
    """Decoder-only causal transformer policy for Pokemon battles.

    This is the main model class implementing a Metamon-style architecture
    with ~200M parameters for human-level Pokemon battle AI.

    Features:
    - Decoder-only causal transformer
    - RoPE positional encoding
    - SwiGLU activation
    - HL-Gauss value classification head
    - Gradient checkpointing support
    - KV cache for efficient inference
    """

    def __init__(self, config: ScaledEncoderConfig):
        super().__init__()
        self.config = config

        # Input embedding projection
        # This projects the encoded observation to d_model
        # The actual embedding happens in the state encoder
        self.input_proj = nn.Linear(config.d_model, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
                use_flash_attention=config.use_flash_attention,
            )
            for _ in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Policy head (action logits)
        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.ReLU(),
            nn.Linear(512, config.action_dim),
        )

        # Value head will be added separately (HLGaussValueHead)
        # This allows flexibility in value estimation approach
        self.value_head: Optional[nn.Module] = None

        # Configuration
        self.use_checkpointing = config.use_gradient_checkpointing

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with scaled normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def set_value_head(self, value_head: nn.Module):
        """Set the value head module (e.g., HLGaussValueHead)."""
        self.value_head = value_head

    def forward(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """Forward pass through the decoder.

        Args:
            x: Encoded observations [batch, seq_len, d_model]
            action_mask: Boolean mask for valid actions [batch, action_dim]
                         True = valid, False = invalid
            kv_cache: List of KV caches from previous forward passes
            use_cache: Whether to return updated KV cache

        Returns:
            Tuple of:
            - action_probs: Action probabilities [batch, action_dim]
            - value: Value estimate [batch] (or None if no value head)
            - new_kv_cache: Updated KV cache (or None if not caching)
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)

        # Process through transformer blocks
        new_cache = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            block_cache = kv_cache[i] if kv_cache is not None else None

            if self.training and self.use_checkpointing and not use_cache:
                # Gradient checkpointing during training
                x, layer_cache = checkpoint(
                    block, x, block_cache, use_cache,
                    use_reentrant=False
                )
            else:
                x, layer_cache = block(x, block_cache, use_cache)

            if use_cache:
                new_cache.append(layer_cache)

        # Final norm
        x = self.final_norm(x)

        # Use last token for prediction (causal model)
        last_hidden = x[:, -1, :]  # [batch, d_model]

        # Policy output
        logits = self.policy_head(last_hidden)  # [batch, action_dim]

        # Apply action mask
        if action_mask is not None:
            # Mask invalid actions to -inf before softmax
            logits = logits.masked_fill(~action_mask, float("-inf"))

        action_probs = F.softmax(logits, dim=-1)

        # Value output
        value = None
        if self.value_head is not None:
            value = self.value_head(last_hidden)

        return action_probs, value, new_cache

    def get_action(
        self,
        x: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        kv_cache: Optional[list] = None,
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[list]]:
        """Select an action from the policy.

        Args:
            x: Encoded observation [batch, seq_len, d_model]
            action_mask: Valid action mask [batch, action_dim]
            deterministic: If True, select argmax action
            kv_cache: Previous KV cache for incremental decoding

        Returns:
            Tuple of (action_index, log_prob, value, new_kv_cache)
        """
        action_probs, value, new_cache = self.forward(
            x, action_mask, kv_cache, use_cache=True
        )

        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

        # Get log probability
        log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)) + 1e-8)
        log_prob = log_prob.squeeze(-1)

        return action.item(), log_prob, value, new_cache

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_decoder_policy(config: Optional[ScaledEncoderConfig] = None) -> DecoderPolicy:
    """Factory function to create a DecoderPolicy.

    Args:
        config: Model configuration. Defaults to LARGE_CONFIG (~200M params).

    Returns:
        Configured DecoderPolicy instance.
    """
    from .config import LARGE_CONFIG

    if config is None:
        config = LARGE_CONFIG

    return DecoderPolicy(config)
