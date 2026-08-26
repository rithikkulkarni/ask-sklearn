"""Turn raw issue+comment JSON (data/raw/) into thread-aware chunks
(data/processed/chunks.jsonl), one JSON object per chunk.

Usage:
    python -m src.chunking.chunk_issues
"""

import json
import re
from pathlib import Path

from src.chunking.tokenizer import (
    count_tokens,
    get_encoding,
    split_by_tokens,
    take_last_tokens,
)
from src.config import load_config

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.jsonl")

# Matches mentions like "scikit-learn 1.3.2", "sklearn==1.2.0", "sklearn: 1.3.2".
VERSION_RE = re.compile(
    r"(?:scikit-learn|sklearn)[\s:=]*v?(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE
)


def load_issues():
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def extract_component(issue, module_prefix):
    """Component = first `module:X` label on the issue, if any."""
    for label in issue["labels"]:
        if label.startswith(module_prefix):
            return label[len(module_prefix) :]
    return None


def extract_version(issue):
    """Best-effort scikit-learn version mention: checked in the body first, then comments."""
    texts = [issue["body"] or ""] + [
        comment["body"] or "" for comment in issue["comments"]
    ]
    for text in texts:
        match = VERSION_RE.search(text)
        if match:
            return match.group(1)
    return None


def build_units(issue):
    """Ordered list of (text, source_metadata) covering the issue body then each comment."""
    units = [(issue["body"] or "", {"type": "body"})]
    for i, comment in enumerate(issue["comments"]):
        units.append(
            (
                comment["body"] or "",
                {
                    "type": "comment",
                    "comment_index": i,
                    "author": comment["author"],
                    "created_at": comment["created_at"],
                },
            )
        )
    return units


def chunk_issue(issue, encoding, chunk_size, overlap):
    """Thread-aware windowed chunking: accumulate units until the token budget is hit,
    then start a new chunk, carrying forward a token-overlap tail from the previous
    chunk for continuity. Every chunk is prefixed with the issue number + title so it
    stays self-contained even if it's mostly a later comment.
    """
    prefix = f"Issue #{issue['number']}: {issue['title']}\n\n"
    prefix_tokens = count_tokens(prefix, encoding)
    budget = chunk_size - prefix_tokens
    if budget <= 0:
        raise ValueError("chunk_size_tokens too small to fit the issue-title prefix")

    chunks = []
    body_parts, meta_parts = [], []

    def joined_tokens(parts):
        return count_tokens("\n\n".join(parts), encoding) if parts else 0

    def flush():
        nonlocal body_parts, meta_parts
        if not body_parts:
            return
        body_text = "\n\n".join(body_parts)
        chunks.append((prefix + body_text, meta_parts))

        tail = take_last_tokens(body_text, encoding, overlap)
        # Re-measure the tail on its own, since decoding a token slice back to text
        # and re-encoding it can land on a different token count than requested.
        if tail and count_tokens(tail, encoding) <= budget:
            body_parts = [tail]
            meta_parts = [{"type": "context_carry"}]
        else:
            body_parts, meta_parts = [], []

    for unit_text, unit_meta in build_units(issue):
        if not unit_text.strip():
            continue

        if count_tokens(unit_text, encoding) > budget:
            flush()
            for piece in split_by_tokens(unit_text, encoding, budget, overlap):
                chunks.append((prefix + piece, [unit_meta]))
            continue

        if body_parts and joined_tokens(body_parts + [unit_text]) > budget:
            flush()
            # The carried-forward tail plus this unit can still overflow (e.g. a
            # token-inefficient block like a stack trace). Since the unit alone
            # already passed the budget check above, drop the tail rather than
            # risk another overflow.
            if body_parts and joined_tokens(body_parts + [unit_text]) > budget:
                body_parts, meta_parts = [], []

        body_parts.append(unit_text)
        meta_parts.append(unit_meta)

    flush()
    return chunks


def main():
    config = load_config()
    chunking_cfg = config.chunking
    encoding = get_encoding(chunking_cfg.tokenizer_encoding)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_issues = 0
    total_chunks = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for issue in load_issues():
            component = extract_component(issue, chunking_cfg.module_label_prefix)
            version = extract_version(issue)

            chunks = chunk_issue(
                issue,
                encoding,
                chunking_cfg.chunk_size_tokens,
                chunking_cfg.chunk_overlap_tokens,
            )

            for i, (text, sources) in enumerate(chunks):
                record = {
                    "chunk_id": f"{issue['number']}-{i}",
                    "issue_number": issue["number"],
                    "chunk_index": i,
                    "text": text,
                    "token_count": count_tokens(text, encoding),
                    "component": component,
                    "version": version,
                    "sources": sources,
                }
                out.write(json.dumps(record) + "\n")
                total_chunks += 1

            total_issues += 1
            if total_issues % 500 == 0:
                print(
                    f"Chunked {total_issues} issues -> {total_chunks} chunks so far..."
                )

    print(f"Done. {total_issues} issues -> {total_chunks} chunks written to {OUT_PATH}")


if __name__ == "__main__":
    main()
