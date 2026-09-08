"""Cross-question rollup math for a single eval run (one retrieval mode's worth
of per-question records, as produced by run_eval.py).
"""


def mean_score(values: list[float | None]) -> float | None:
    """Mean of the non-None values, or None if there are none."""
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def refusal_accuracy(records: list[dict]) -> float | None:
    """Among questions tagged "refusal", the fraction where the system actually
    refused. None if the run had no refusal-tagged questions.
    """
    refusal_records = [r for r in records if "refusal" in r["tags"]]
    if not refusal_records:
        return None
    correct = sum(1 for r in refusal_records if r["refused"])
    return correct / len(refusal_records)


def build_summary(records: list[dict]) -> dict:
    """Aggregate a mode's per-question records into a single summary dict."""
    retrieval_metrics = [r["retrieval_metrics"] for r in records]
    groundedness_scores = [
        r["groundedness"]["score"] if r["groundedness"] else None for r in records
    ]

    return {
        "n_questions": len(records),
        "mean_precision": mean_score(
            [m["precision"] for m in retrieval_metrics if m is not None]
        ),
        "mean_recall": mean_score(
            [m["recall"] for m in retrieval_metrics if m is not None]
        ),
        "mean_citation_validity_rate": mean_score(
            [r["citation_validity_rate"] for r in records]
        ),
        "mean_groundedness": mean_score(groundedness_scores),
        "mean_correctness": mean_score([r["correctness"]["score"] for r in records]),
        "refusal_accuracy": refusal_accuracy(records),
    }
