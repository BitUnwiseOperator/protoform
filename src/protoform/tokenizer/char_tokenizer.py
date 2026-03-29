"""Character-level tokenizer for the Shakespeare dataset."""


class CharTokenizer:
    """A character-level tokenizer that maps each unique character to an integer.

    The vocabulary is built from the characters present in the corpus passed
    at construction time. Characters are sorted so the mapping is stable and
    deterministic across runs.

    Args:
        corpus: The full text to build the vocabulary from.

    Example:
        >>> from protoform.data.shakespeare import load_split
        >>> tok = CharTokenizer(load_split("train"))
        >>> tok.encode("First")
        [19, 47, 57, 58, 59]
        >>> tok.decode(tok.encode("First"))
        'First'
    """

    def __init__(self, corpus: str) -> None:
        """Build vocabulary from all unique characters in the corpus."""
        self._chars: list[str] = sorted(set(corpus))
        self._stoi: dict[str, int] = {c: i for i, c in enumerate(self._chars)}

    @property
    def vocab_size(self) -> int:
        """Number of unique characters in the vocabulary."""
        return len(self._chars)

    def encode(self, data: str) -> list[int]:
        """Convert a string into a list of token IDs.

        Args:
            data: The string to encode.

        Returns:
            A list of integer token IDs.

        Raises:
            ValueError: If any character in data is not in the vocabulary.
        """
        ids: list[int] = []
        for char in data:
            if char not in self._stoi:
                raise ValueError(
                    f"'{char}' is not in vocab "
                    f"-- was the tokenizer built from the right corpus?"
                )
            ids.append(self._stoi[char])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert a list of token IDs back into a string.

        Args:
            ids: The list of integer token IDs to decode.

        Returns:
            The decoded string.

        Raises:
            ValueError: If any ID is outside the vocabulary range.
        """
        chars: list[str] = []
        for i in ids:
            if not (0 <= i < len(self._chars)):
                raise ValueError(
                    f"{i} is not a valid token ID "
                    f"-- valid range is 0-{len(self._chars) - 1}"
                )
            chars.append(self._chars[i])
        return "".join(chars)
