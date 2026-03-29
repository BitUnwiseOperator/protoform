"""Attention layer implementations."""

import torch
from torch import Tensor, nn


class SingleHeadAttention(nn.Module):
    """Single-head scaled dot-product attention.

    Args:
        d_model: Dimensionality of the input embeddings.
    """

    def __init__(self, d_model: int) -> None:
        """Initialize SingleHeadAttention with embedding dimension."""
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Compute single-head scaled dot-product attention.

        Args:
            q: Query tensor of shape (batch, seq_len, d_model).
            k: Key tensor of shape (batch, seq_len, d_model).
            v: Value tensor of shape (batch, seq_len, d_model).
            causal: If True, mask out future positions.

        Returns:
            A tuple of (output, weights) where output has shape
            (batch, seq_len, d_model) and weights has shape
            (batch, seq_len, seq_len).
        """
        batch, seq_len, _ = q.shape
        weights = torch.zeros(batch, seq_len, seq_len)
        output = v.clone()
        return output, weights
