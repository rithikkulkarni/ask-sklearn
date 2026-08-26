import json

from src.chunking.chunk_issues import (
    build_units,
    chunk_issue,
    extract_component,
    extract_version,
    load_issues,
)
from tests.fakes import FakeEncoding


def test_extract_component_returns_first_matching_label():
    issue = {"labels": ["bug", "module:model_selection", "module:ensemble"]}
    assert extract_component(issue, "module:") == "model_selection"


def test_extract_component_returns_none_when_no_match():
    issue = {"labels": ["bug", "help wanted"]}
    assert extract_component(issue, "module:") is None


def test_extract_version_finds_mention_in_body():
    issue = {"body": "Happens on scikit-learn 1.3.2 for me.", "comments": []}
    assert extract_version(issue) == "1.3.2"


def test_extract_version_falls_back_to_comments():
    issue = {
        "body": "No version mentioned here.",
        "comments": [{"body": "I'm on sklearn==1.2.0"}],
    }
    assert extract_version(issue) == "1.2.0"


def test_extract_version_returns_none_when_absent():
    issue = {"body": "No version anywhere.", "comments": [{"body": "me neither"}]}
    assert extract_version(issue) is None


def test_build_units_orders_body_then_comments():
    issue = {
        "body": "the body text",
        "comments": [
            {"body": "first comment", "author": "alice", "created_at": "t1"},
            {"body": "second comment", "author": "bob", "created_at": "t2"},
        ],
    }

    units = build_units(issue)

    assert [text for text, _ in units] == [
        "the body text",
        "first comment",
        "second comment",
    ]
    assert units[0][1] == {"type": "body"}
    assert units[1][1] == {
        "type": "comment",
        "comment_index": 0,
        "author": "alice",
        "created_at": "t1",
    }
    assert units[2][1] == {
        "type": "comment",
        "comment_index": 1,
        "author": "bob",
        "created_at": "t2",
    }


def test_chunk_issue_fits_everything_in_one_chunk_when_small():
    issue = {
        "number": 1,
        "title": "short",
        "body": "small body",
        "comments": [{"body": "small comment", "author": "a", "created_at": "t"}],
    }

    chunks = chunk_issue(issue, FakeEncoding(), chunk_size=50, overlap=2)

    assert len(chunks) == 1
    text, sources = chunks[0]
    assert text.startswith("Issue #1: short\n\n")
    assert "small body" in text
    assert "small comment" in text
    assert sources[0] == {"type": "body"}


def test_chunk_issue_splits_oversized_unit_with_carried_prefix():
    issue = {
        "number": 1,
        "title": "t",
        "body": "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10",
        "comments": [],
    }

    chunks = chunk_issue(issue, FakeEncoding(), chunk_size=8, overlap=2)

    assert len(chunks) > 1
    for text, _ in chunks:
        assert text.startswith("Issue #1: t\n\n")


def test_load_issues_reads_sorted_shards(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "issues_shard_0001.jsonl").write_text(
        json.dumps({"number": 2}) + "\n", encoding="utf-8"
    )
    (raw_dir / "issues_shard_0000.jsonl").write_text(
        json.dumps({"number": 1}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr("src.chunking.chunk_issues.RAW_DIR", raw_dir)

    issues = list(load_issues())

    assert [issue["number"] for issue in issues] == [1, 2]
