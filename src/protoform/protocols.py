"""Structural interfaces for the protoform library.

All public-facing functions type against one of these Protocols rather than
a concrete class.  This keeps modules decoupled — any object that satisfies
the interface works, regardless of where it comes from.

The two core abstractions are:

- ``Corpus`` / ``TextCorpus`` — what trainable data looks like.
- ``Tokenizer[CorpusT]`` — anything that can convert a ``Corpus`` type
  ``CorpusT`` into integer sequences and back.

Because ``Tokenizer`` is generic over ``CorpusT``, the type checker enforces
that the corpus and the tokenizer agree on modality:

    tok: Tokenizer[str]           # a text tokenizer
    tok: Tokenizer[ImageCorpus]   # an image tokenizer (future)
    tok: Tokenizer[AudioCorpus]   # an audio tokenizer (future)

Passing an audio tokenizer a text corpus is a type error, not a runtime crash.
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


class Corpus(Protocol):
    """Base contract for any dataset that can be tokenized.

    The only universal requirement across all modalities is that a corpus
    has a measurable length — needed for split boundary arithmetic.

    ``str``, ``list``, numpy arrays, and memory-mapped files all satisfy
    this protocol out of the box.
    """

    def __len__(self) -> int:
        """Number of top-level elements (characters, frames, patches, …)."""
        ...


class TextCorpus(Corpus, Protocol):
    """A corpus of text — iterable over characters, sliceable by position.

    ``str`` satisfies this protocol without any changes, so existing code
    that passes a plain string continues to work.

    A future implementation could back this with ``mmap`` for zero-copy
    slicing — the tokenizer would not need to change.
    """

    def __iter__(self) -> Iterator[str]:
        """Iterate over individual characters."""
        ...

    def __getitem__(self, index: slice) -> str:
        """Return a contiguous slice of the text."""
        ...


@runtime_checkable
class Tokenizer[CorpusT: Corpus](Protocol):
    """Converts a corpus of type ``CorpusT`` into integer sequences and back.

    Generic over the corpus type so the type checker can enforce that the
    tokenizer and its input agree on modality.

    ``CharTokenizer`` satisfies ``Tokenizer[str]`` structurally — it needs
    no changes and no explicit inheritance.

    Example:
        >>> from protoform.tokenizer import CharTokenizer
        >>> from protoform.protocols import Tokenizer
        >>> tok = CharTokenizer("hello world")
        >>> isinstance(tok, Tokenizer)
        True
    """

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        ...

    def encode(self, data: CorpusT) -> list[int]:
        """Convert a corpus element into a list of integer token IDs.

        Returns ``list[int]`` because transformers are discrete by design —
        tokens are indices into a vocabulary or codebook looked up via
        ``nn.Embedding``.  This holds across all modalities:

        - Text (char-level, BPE): character / subword → codebook index
        - Images (VQ-VAE, VQGAN): patch → codebook index
        - Audio (Encodec): frame → codebook index

        Note:
            A continuous encoder (CLIP, BERT) that produces float embeddings
            directly is a different abstraction — an *encoder*, not a
            *tokenizer*.  A future stage of ``protoform`` will extend this
            protocol to ``Tokenizer[CorpusT, TokenT]``, adding a second type
            parameter for the token type and covering continuous modalities.
        """
        ...

    def decode(self, ids: list[int]) -> CorpusT:
        """Convert a list of integer token IDs back into a corpus element."""
        ...
