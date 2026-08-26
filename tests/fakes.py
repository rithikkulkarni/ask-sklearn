"""Test doubles shared across the test suite."""


class FakeEncoding:
    """Whitespace-based fake tokenizer: each token is one whitespace-split word.

    Used instead of real tiktoken in chunking tests so window/overlap math is exact
    and human-checkable, and the suite never needs network access to fetch a vocab.
    """

    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)
