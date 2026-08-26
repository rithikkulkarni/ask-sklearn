import json
from unittest.mock import patch

from src.ingest.fetch_issues import (
    build_record,
    fetch_all_comments_index,
    fetch_all_issues,
    flush_shard,
)


def test_fetch_all_issues_filters_out_pull_requests():
    items = [
        {"number": 1, "title": "issue one"},
        {"number": 2, "title": "a pr", "pull_request": {"url": "..."}},
        {"number": 3, "title": "issue three"},
    ]

    with patch("src.ingest.fetch_issues.paginated_get", return_value=iter(items)):
        result = list(fetch_all_issues(session=object()))

    assert [item["number"] for item in result] == [1, 3]


def test_fetch_all_comments_index_groups_by_issue_number():
    comments = [
        {
            "issue_url": "https://api.github.com/repos/scikit-learn/scikit-learn/issues/42",
            "user": {"login": "alice"},
            "body": "first comment",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "issue_url": "https://api.github.com/repos/scikit-learn/scikit-learn/issues/42",
            "user": None,
            "body": "second comment",
            "created_at": "2026-01-02T00:00:00Z",
        },
        {
            "issue_url": "https://api.github.com/repos/scikit-learn/scikit-learn/issues/7",
            "user": {"login": "bob"},
            "body": "other issue's comment",
            "created_at": "2026-01-03T00:00:00Z",
        },
    ]

    with patch("src.ingest.fetch_issues.paginated_get", return_value=iter(comments)):
        result = fetch_all_comments_index(session=object())

    assert list(result.keys()) == [42, 7]
    assert result[42] == [
        {
            "author": "alice",
            "body": "first comment",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "author": None,
            "body": "second comment",
            "created_at": "2026-01-02T00:00:00Z",
        },
    ]
    assert result[7] == [
        {
            "author": "bob",
            "body": "other issue's comment",
            "created_at": "2026-01-03T00:00:00Z",
        }
    ]


def test_build_record_shape():
    issue = {
        "number": 5,
        "title": "Bug in GridSearchCV",
        "body": "It crashes",
        "state": "closed",
        "labels": [{"name": "bug"}, {"name": "module:model_selection"}],
        "created_at": "2026-01-01T00:00:00Z",
        "closed_at": "2026-01-05T00:00:00Z",
    }
    comments_by_issue = {
        5: [{"author": "alice", "body": "me too", "created_at": "2026-01-02T00:00:00Z"}]
    }

    record = build_record(issue, comments_by_issue)

    assert record == {
        "number": 5,
        "title": "Bug in GridSearchCV",
        "body": "It crashes",
        "state": "closed",
        "labels": ["bug", "module:model_selection"],
        "created_at": "2026-01-01T00:00:00Z",
        "closed_at": "2026-01-05T00:00:00Z",
        "comments": [
            {"author": "alice", "body": "me too", "created_at": "2026-01-02T00:00:00Z"}
        ],
    }


def test_build_record_defaults_to_empty_comments():
    issue = {
        "number": 6,
        "title": "No comments yet",
        "body": None,
        "state": "open",
        "labels": [],
        "created_at": "2026-01-01T00:00:00Z",
        "closed_at": None,
    }

    record = build_record(issue, comments_by_issue={})

    assert record["comments"] == []


def test_flush_shard_writes_jsonl(tmp_path, monkeypatch):
    out_dir = tmp_path / "raw"
    monkeypatch.setattr("src.ingest.fetch_issues.OUT_DIR", out_dir)

    records = [{"number": 1}, {"number": 2}]
    flush_shard(records, shard_index=0)

    shard_path = out_dir / "issues_shard_0000.jsonl"
    lines = shard_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == records
