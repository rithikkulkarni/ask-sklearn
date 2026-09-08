"""Retrieval precision/recall, computed at the issue level.

Ground truth (data/eval/test_set.json) targets whole issues, not chunks, since
that's stable across chunking changes -- see the eval-harness ticket. Retrieval
returns chunks, so `retrieved_issue_numbers` is expected to already be a
deduped set/list of the issue numbers behind the retrieved chunks (see
run_eval.py, which dedupes before calling this).
"""


def precision_recall(
    retrieved_issue_numbers: list[int], target_issue_numbers: list[int]
) -> dict | None:
    """Precision/recall of a retrieval run against a question's target issues.

    Returns None if the question has no target issues (e.g. an out-of-domain
    refusal question), since precision/recall aren't meaningful there.
    """
    if not target_issue_numbers:
        return None

    retrieved = set(retrieved_issue_numbers)
    target = set(target_issue_numbers)
    true_positives = len(retrieved & target)

    precision = true_positives / len(retrieved) if retrieved else 0.0
    recall = true_positives / len(target)

    return {"precision": precision, "recall": recall}
