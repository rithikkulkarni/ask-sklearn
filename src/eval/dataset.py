"""Loads and validates the hand-written eval question set
(data/eval/test_set.json).
"""

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("data/eval/test_set.json")


@dataclass
class EvalQuestion:
    id: str
    query: str
    target_issue_numbers: list[int]
    rubric_facts: list[str]
    tags: list[str]
    smoke: bool


def load_questions(
    path: Path = DEFAULT_PATH, smoke_only: bool = False
) -> list[EvalQuestion]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    questions = [
        EvalQuestion(
            id=item["id"],
            query=item["query"],
            target_issue_numbers=item["target_issue_numbers"],
            rubric_facts=item["rubric_facts"],
            tags=item.get("tags", []),
            smoke=item["smoke"],
        )
        for item in raw
    ]

    if smoke_only:
        questions = [q for q in questions if q.smoke]

    return questions
