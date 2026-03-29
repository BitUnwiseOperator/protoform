"""Tests that any Attention implementation must satisfy."""

# torch stubs lack full type info for manual_seed
# pyright: reportUnknownMemberType=none

import pytest
import torch

from protoform.layers.attention import SingleHeadAttention


class TestAttentionProtocol:
    """Tests for the Attention protocol contract.

    Any class that implements Attention must:
    1. Accept (Q, K, V) tensors and return (output, weights)
    2. Preserve sequence length and embedding dim in output
    3. Produce weights that sum to 1 along the key dimension
    4. Respect a causal mask (future positions get zero weight)
    """

    def setup_method(self) -> None:
        """Set up shared Q, K, V tensors for each test."""
        torch.manual_seed(1998)
        self.batch = 2
        self.seq_len = 4
        self.d_model = 8
        self.query = torch.randn(self.batch, self.seq_len, self.d_model)
        self.key = torch.randn(self.batch, self.seq_len, self.d_model)
        self.value = torch.randn(self.batch, self.seq_len, self.d_model)

    def test_output_shape_matches_input(self) -> None:
        """Output tensor must have same shape as input: (batch, seq_len, d_model)."""
        attn = SingleHeadAttention(d_model=self.d_model)
        output, _weights = attn(self.query, self.key, self.value)
        assert output.shape == (self.batch, self.seq_len, self.d_model)

    def test_weights_shape(self) -> None:
        """Weights must be (batch, seq_len, seq_len) — one score per query-key pair."""
        attn = SingleHeadAttention(d_model=self.d_model)
        _output, weights = attn(self.query, self.key, self.value)
        assert weights.shape == (self.batch, self.seq_len, self.seq_len)

    @pytest.mark.xfail(reason="stub implementation returns zeros; needs real softmax")
    def test_weights_sum_to_one(self) -> None:
        """After softmax, each query's weights over all keys must sum to 1."""
        attn = SingleHeadAttention(d_model=self.d_model)
        _output, weights = attn(self.query, self.key, self.value)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_causal_mask_zeroes_future(self) -> None:
        """With causal=True, position i must not attend to positions j > i."""
        attn = SingleHeadAttention(d_model=self.d_model)
        _output, weights = attn(self.query, self.key, self.value, causal=True)

        # upper triangle (future positions) should be zero
        for b in range(self.batch):
            for i in range(self.seq_len):
                for j in range(i + 1, self.seq_len):
                    assert weights[b, i, j].item() == 0.0

    @pytest.mark.xfail(reason="stub implementation returns zeros; needs real softmax")
    def test_causal_weights_still_sum_to_one(self) -> None:
        """Even with masking, the visible weights must still sum to 1."""
        attn = SingleHeadAttention(d_model=self.d_model)
        _output, weights = attn(self.query, self.key, self.value, causal=True)
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_deterministic_with_seed(self) -> None:
        """Same seed must produce identical output."""
        attn = SingleHeadAttention(d_model=self.d_model)

        torch.manual_seed(1998)
        q = torch.randn(1, 3, self.d_model)
        k = torch.randn(1, 3, self.d_model)
        v = torch.randn(1, 3, self.d_model)
        out_a, _ = attn(q, k, v)

        torch.manual_seed(1998)
        q = torch.randn(1, 3, self.d_model)
        k = torch.randn(1, 3, self.d_model)
        v = torch.randn(1, 3, self.d_model)
        out_b, _ = attn(q, k, v)

        assert torch.equal(out_a, out_b)
