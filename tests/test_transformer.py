"""Tests for the base Transformer model."""

from protoform.models.transformer import Transformer


def test_transformer() -> None:
    """Transformer stores d_model and vocab_size on init."""
    t = Transformer(d_model=32, vocab_size=65)
    assert t.d_model == 32
    assert t.vocab_size == 65
