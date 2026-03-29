# pyright: basic
# This file is the HuggingFace adapter boundary. All untyped third-party calls
# (datasets, torch) are contained here. Outputs are immediately cast to clean
# primitives (str, Tensor) before being returned to the rest of the codebase.
# Strict mode is intentionally disabled for this file only — every other module
# runs strict.
"""Shakespeare's collected works as a training dataset.

Fetched automatically from the Hugging Face Hub on first use — no local
file required.  The dataset is cached by the ``datasets`` library under
``~/.cache/huggingface/datasets/`` so subsequent calls are instant.

Three functions are provided:

- ``load_corpus`` — returns the full raw text.  Use this to inspect the
  data or as a ground-truth oracle in tests.
- ``load_split`` — returns one named piece of the corpus (train /
  validation / test), used for building the tokenizer vocabulary.
- ``load_encoded`` — encodes the full corpus once and returns integer
  tensors ready for training.  This is the efficient path: one network
  fetch, one encode pass, then split by index.

Example::

    from protoform.data.shakespeare import load_corpus, load_encoded, load_split
    from protoform.tokenizer import CharTokenizer

    tok = CharTokenizer(load_split("train"))
    splits = load_encoded(tok)
    splits["train"]       # torch.Tensor of shape (1_003_854,)
    splits["validation"]
    splits["test"]
"""

import torch
from datasets import load_dataset
from torch import Tensor

from protoform.protocols import Tokenizer

_DATASET_REPO: str = "transformingit/tiny-shakespeare"

SPLIT_FRACTIONS: dict[str, tuple[float, float]] = {
    "train": (0.0, 0.9),
    "validation": (0.9, 0.95),
    "test": (0.95, 1.0),
}


def load_corpus() -> str:
    """Fetch the full Shakespeare corpus from the Hugging Face Hub.

    The ``datasets`` library caches the result locally after the first
    download, so this is a network call only once per machine.

    Returns:
        The complete corpus as a single plain string.

    Example:
        >>> text = load_corpus()
        >>> text[:14]
        'First Citizen:'
    """
    ds = load_dataset(_DATASET_REPO, split="train")
    return str(ds[0]["text"])


def load_split(split: str) -> str:
    """Return one piece of the Shakespeare corpus as plain text.

    The corpus is cut into three non-overlapping pieces by position.
    The model trains on ``"train"``, we tune on ``"validation"``, and
    we measure final performance on ``"test"``.

    Args:
        split: Which piece to return.  One of ``"train"`` (90 %),
            ``"validation"`` (5 %), or ``"test"`` (5 %).

    Returns:
        The requested piece as a plain string.

    Raises:
        ValueError: If ``split`` is not one of the three valid names.

    Example:
        >>> train = load_split("train")
        >>> train[:14]
        'First Citizen:'
        >>> len(load_split("validation")) < len(load_split("train"))
        True
    """
    if split not in SPLIT_FRACTIONS:
        raise ValueError(
            f"split must be one of {sorted(SPLIT_FRACTIONS)}, got {split!r}"
        )
    text = load_corpus()
    n = len(text)
    s, e = SPLIT_FRACTIONS[split]
    return text[round(n * s) : round(n * e)]


def load_encoded(tokenizer: Tokenizer[str]) -> dict[str, Tensor]:
    """Encode the full corpus once and return split tensors.

    Fetches the corpus once, encodes every character into an integer,
    then splits the resulting tensor by index.  All training code should
    use this function rather than ``load_split`` to avoid re-encoding on
    every access.

    Args:
        tokenizer: Any object satisfying ``Tokenizer[str]`` — encodes a
            string into token IDs and decodes them back.

    Returns:
        A dict with keys ``"train"``, ``"validation"``, and ``"test"``,
        each mapping to a 1-D ``torch.long`` tensor of token IDs.

    Example:
        >>> from protoform.tokenizer import CharTokenizer
        >>> from protoform.data.shakespeare import load_split, load_encoded
        >>> tok = CharTokenizer(load_split("train"))
        >>> splits = load_encoded(tok)
        >>> splits["train"].shape
        torch.Size([1003854])
    """
    data = torch.tensor(tokenizer.encode(load_corpus()), dtype=torch.long)
    n = len(data)
    return {
        name: data[round(n * s) : round(n * e)]
        for name, (s, e) in SPLIT_FRACTIONS.items()
    }
