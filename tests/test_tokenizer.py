"""Tests for the character-level tokenizer."""

import random

import pytest

from protoform.tokenizer.char_tokenizer import CharTokenizer

# Fixtures (tokenizer, text_splits) are provided by conftest.py.


def test_encode_decode_roundtrip(tokenizer: CharTokenizer) -> None:
    """Encoding then decoding must return the original string."""
    text = "First Citizen:"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_same_character_same_id(tokenizer: CharTokenizer) -> None:
    """The same character must always map to the same token ID."""
    ids = tokenizer.encode("rrrr")
    assert len(set(ids)) == 1


def test_vocab_size_matches_unique_chars(text_splits: dict[str, str]) -> None:
    """vocab_size must equal the number of unique characters in the training corpus."""
    train = text_splits["train"]
    tok = CharTokenizer(train)
    assert tok.vocab_size == len(set(train))


def test_unknown_character_raises() -> None:
    """Encoding a character not in the vocab must raise ValueError."""
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError, match="not in vocab"):
        tok.encode("z")


def test_invalid_token_id_raises() -> None:
    """Decoding an ID outside the vocabulary range must raise ValueError."""
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError, match="not a valid token ID"):
        tok.decode([999])


def test_random_roundtrips(text_splits: dict[str, str]) -> None:
    """1000 random slices of the training split must roundtrip through encode/decode."""
    random.seed(1998)
    train = text_splits["train"]
    tok = CharTokenizer(train)
    n = len(train)

    for _ in range(1000):
        start = random.randint(0, n - 2)
        end = random.randint(start + 1, min(start + 200, n))
        chunk = train[start:end]
        assert tok.decode(tok.encode(chunk)) == chunk
