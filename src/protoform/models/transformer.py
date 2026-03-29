"""Base Transformer model for protoform tutorial stages."""


class Transformer:
    """A base transformer model.

    Args:
        d_model: The dimensionality of the token embeddings.
        vocab_size: The number of unique tokens in the vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """Initialize a Transformer with embedding dimension and vocabulary size."""
        self.d_model = d_model
        self.vocab_size = vocab_size
