"""Tests for the Shakespeare data loader."""

from typing import cast

import pytest
import torch
from torch import Tensor

from protoform.data.shakespeare import SPLIT_FRACTIONS, load_split
from protoform.tokenizer import CharTokenizer

# Fixtures (corpus, text_splits, tokenizer, encoded) are provided by conftest.py.

# ── Split correctness ─────────────────────────────────────────────────────────


def test_splits_cover_full_corpus(corpus: str, text_splits: dict[str, str]) -> None:
    """All three splits combined must equal the full corpus length."""
    total = sum(len(s) for s in text_splits.values())
    assert total == len(corpus)


def test_splits_are_contiguous(corpus: str, text_splits: dict[str, str]) -> None:
    """Each split's length must match its declared fraction of the corpus."""
    n = len(corpus)
    for name, (s, e) in SPLIT_FRACTIONS.items():
        assert len(text_splits[name]) == round(n * e) - round(n * s)


def test_invalid_split_raises() -> None:
    """Requesting an unknown split name must raise ValueError."""
    with pytest.raises(ValueError, match="split must be one of"):
        load_split("garbage")


def test_train_is_largest_split(text_splits: dict[str, str]) -> None:
    """Train must be larger than validation and test."""
    assert len(text_splits["train"]) > len(text_splits["validation"])
    assert len(text_splits["train"]) > len(text_splits["test"])


def test_splits_reconstruct_full_corpus(
    corpus: str, text_splits: dict[str, str]
) -> None:
    """Concatenating all splits in order must reproduce the full corpus exactly."""
    reconstructed = (
        text_splits["train"] + text_splits["validation"] + text_splits["test"]
    )
    assert reconstructed == corpus


# ── Encoded tensors ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("split_name", ["train", "validation", "test"])
def test_load_encoded_returns_tensors(
    split_name: str, encoded: dict[str, Tensor]
) -> None:
    """load_encoded must return a dict of torch.long tensors."""
    assert split_name in encoded
    assert isinstance(encoded[split_name], torch.Tensor)
    assert encoded[split_name].dtype == torch.long


def test_load_encoded_total_length_matches_corpus(
    corpus: str, encoded: dict[str, Tensor]
) -> None:
    """Encoded splits combined must account for every character in the corpus."""
    total = sum(len(s) for s in encoded.values())
    assert total == len(corpus)


def test_load_encoded_decodes_back_to_text(
    corpus: str,
    tokenizer: CharTokenizer,
    encoded: dict[str, Tensor],
) -> None:
    """Decoding encoded splits must reproduce the original corpus exactly."""
    # Tensor.tolist() is partially typed in torch's stubs — the method signature
    # is inferred as () -> list[Unknown]. cast() asserts the type we know is
    # correct: load_encoded always uses dtype=torch.long.
    ids = (
        cast(list[int], encoded["train"].tolist())  # pyright: ignore[reportUnknownMemberType]
        + cast(list[int], encoded["validation"].tolist())  # pyright: ignore[reportUnknownMemberType]
        + cast(list[int], encoded["test"].tolist())  # pyright: ignore[reportUnknownMemberType]
    )
    assert tokenizer.decode(ids) == corpus
