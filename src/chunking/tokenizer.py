"""tiktoken wrapper for token counting and token-window splitting."""

import tiktoken

_encoding_cache = {}


def get_encoding(name: str):
    if name not in _encoding_cache:
        _encoding_cache[name] = tiktoken.get_encoding(name)
    return _encoding_cache[name]


def count_tokens(text: str, encoding) -> int:
    return len(encoding.encode(text))


def take_last_tokens(text: str, encoding, n: int) -> str:
    """Return the trailing `n` tokens of `text`, decoded back to a string."""
    if n <= 0:
        return ""
    tokens = encoding.encode(text)
    return encoding.decode(tokens[-n:]) if tokens else ""


def split_by_tokens(text: str, encoding, window: int, overlap: int):
    """Split text into overlapping windows of at most `window` tokens each.

    Used only as a fallback for a single unit (e.g. one comment) that alone
    exceeds the chunk token budget.
    """
    tokens = encoding.encode(text)
    if not tokens:
        return []

    step = max(window - overlap, 1)
    pieces = []
    for start in range(0, len(tokens), step):
        piece_tokens = tokens[start : start + window]
        pieces.append(encoding.decode(piece_tokens))
        if start + window >= len(tokens):
            break
    return pieces
