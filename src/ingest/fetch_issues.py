"""Pull scikit-learn/scikit-learn issues + comments from GitHub and store them as
sharded JSONL files under data/raw/.

Usage:
    python -m src.ingest.fetch_issues
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from src.ingest.github_client import API_ROOT, make_session, paginated_get

OWNER = "scikit-learn"
REPO = "scikit-learn"
SHARD_SIZE = 500
OUT_DIR = Path("data/raw")

ISSUE_URL_NUMBER_RE = re.compile(r"/issues/(\d+)$")


def fetch_all_issues(session):
    """Yield raw issue dicts (state=all), skipping pull requests.

    The /issues endpoint returns both issues and PRs; PRs carry a `pull_request` key.
    """
    url = f"{API_ROOT}/repos/{OWNER}/{REPO}/issues"
    params = {"state": "all", "per_page": 100, "sort": "created", "direction": "asc"}
    for item in paginated_get(session, url, params):
        if "pull_request" in item:
            continue
        yield item


def fetch_all_comments_index(session):
    """Fetch every comment on the repo once and index it by issue number.

    Uses the repo-wide comments endpoint instead of one request per issue.
    """
    url = f"{API_ROOT}/repos/{OWNER}/{REPO}/issues/comments"
    params = {"per_page": 100, "sort": "created", "direction": "asc"}

    comments_by_issue = {}
    for comment in paginated_get(session, url, params):
        match = ISSUE_URL_NUMBER_RE.search(comment["issue_url"])
        if not match:
            continue
        issue_number = int(match.group(1))
        comments_by_issue.setdefault(issue_number, []).append(
            {
                "author": comment["user"]["login"] if comment["user"] else None,
                "body": comment["body"],
                "created_at": comment["created_at"],
            }
        )
    return comments_by_issue


def fetch_linked_prs(session, issue_number):
    """Return PRs that cross-reference this issue, via the issue timeline endpoint."""
    url = f"{API_ROOT}/repos/{OWNER}/{REPO}/issues/{issue_number}/timeline"
    linked_prs = []
    for event in paginated_get(session, url, {"per_page": 100}):
        if event.get("event") != "cross-referenced":
            continue
        source_issue = event.get("source", {}).get("issue")
        if source_issue is None or "pull_request" not in source_issue:
            continue
        linked_prs.append(
            {
                "number": source_issue["number"],
                "url": source_issue["html_url"],
            }
        )
    return linked_prs


def build_record(issue, comments_by_issue, session):
    return {
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
        "state": issue["state"],
        "labels": [label["name"] for label in issue["labels"]],
        "created_at": issue["created_at"],
        "closed_at": issue["closed_at"],
        "comments": comments_by_issue.get(issue["number"], []),
        "linked_prs": fetch_linked_prs(session, issue["number"]),
    }


def flush_shard(records, shard_index):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = OUT_DIR / f"issues_shard_{shard_index:04d}.jsonl"
    with open(shard_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)
    print(f"Wrote {len(records)} issues to {shard_path}")


def main():
    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set (add it to .env)")

    session = make_session(token)

    print("Building comments index (repo-wide comments pull)...")
    comments_by_issue = fetch_all_comments_index(session)
    print(f"Indexed comments for {len(comments_by_issue)} issues.")

    shard = []
    shard_index = 0
    total = 0

    for issue in fetch_all_issues(session):
        record = build_record(issue, comments_by_issue, session)
        shard.append(record)
        total += 1

        if total % 100 == 0:
            print(f"Processed {total} issues (issue #{issue['number']})...")

        if len(shard) >= SHARD_SIZE:
            flush_shard(shard, shard_index)
            shard = []
            shard_index += 1

    if shard:
        flush_shard(shard, shard_index)

    print(
        f"Done. {total} issues written across {shard_index + (1 if shard else 0)} shard(s)."
    )


if __name__ == "__main__":
    main()
