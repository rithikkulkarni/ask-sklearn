from unittest.mock import patch

import src.chunking.tokenizer as tokenizer_module
from src.chunking.tokenizer import (
    count_tokens,
    get_encoding,
    split_by_tokens,
    take_last_tokens,
)
from tests.fakes import FakeEncoding


def test_get_encoding_caches_per_name():
    tokenizer_module._encoding_cache.clear()

    with patch("src.chunking.tokenizer.tiktoken.get_encoding") as mock_get_encoding:
        mock_get_encoding.return_value = "fake-encoding-object"

        first = get_encoding("cl100k_base")
        second = get_encoding("cl100k_base")

        assert first == "fake-encoding-object"
        assert second == "fake-encoding-object"
        mock_get_encoding.assert_called_once_with("cl100k_base")


def test_count_tokens_counts_words():
    assert count_tokens("one two three", FakeEncoding()) == 3


def test_take_last_tokens_returns_trailing_window():
    assert take_last_tokens("a b c d e", FakeEncoding(), 2) == "d e"


def test_take_last_tokens_handles_n_le_zero():
    assert take_last_tokens("a b c", FakeEncoding(), 0) == ""


def test_take_last_tokens_handles_empty_text():
    assert take_last_tokens("", FakeEncoding(), 3) == ""


def test_split_by_tokens_windows_with_overlap():
    pieces = split_by_tokens("a b c d e f g", FakeEncoding(), window=3, overlap=1)

    assert pieces == ["a b c", "c d e", "e f g"]


def test_split_by_tokens_empty_text_returns_no_pieces():
    assert split_by_tokens("", FakeEncoding(), window=3, overlap=1) == []
