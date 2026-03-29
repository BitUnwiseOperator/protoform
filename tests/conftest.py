"""Shared pytest fixtures for the protoform test suite.

Fixtures here are available to every test file with no imports required.
All expensive I/O (corpus fetch, encoding) is session-scoped — it runs
exactly once per ``pytest`` invocation regardless of how many test files
or parametrize cases consume it.

Fixture hierarchy
-----------------
corpus          — raw text string (HuggingFace fetch)
  └─ text_splits  — dict[str, str] of the three named text slices
       └─ tokenizer — CharTokenizer built from the training split
            └─ encoded  — dict[str, Tensor] of integer-encoded splits

Each fixture depends only on the one above it, so swapping any layer
(e.g. substituting a tiny synthetic corpus for integration tests) is a
single fixture override in the relevant test file or conftest.
"""

import pytest
from torch import Tensor

from protoform.data.shakespeare import (
    SPLIT_FRACTIONS,
    load_corpus,
    load_encoded,
    load_split,
)
from protoform.tokenizer import CharTokenizer


@pytest.fixture(scope="session")
def corpus() -> str:
    """Full Shakespeare corpus, fetched once per session."""
    return load_corpus()


@pytest.fixture(scope="session")
def text_splits() -> dict[str, str]:
    """All three text splits, loaded once per session."""
    return {name: load_split(name) for name in SPLIT_FRACTIONS}


@pytest.fixture(scope="session")
def tokenizer(text_splits: dict[str, str]) -> CharTokenizer:
    """CharTokenizer built from the training split, constructed once per session."""
    return CharTokenizer(text_splits["train"])


@pytest.fixture(scope="session")
def encoded(tokenizer: CharTokenizer) -> dict[str, Tensor]:
    """Encoded corpus tensors, computed once per session."""
    return load_encoded(tokenizer)
