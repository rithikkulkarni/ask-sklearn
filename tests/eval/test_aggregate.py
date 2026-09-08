from src.eval.aggregate import build_summary, mean_score, refusal_accuracy


def test_mean_score_ignores_none_values():
    assert mean_score([1.0, None, 0.5, None]) == 0.75


def test_mean_score_returns_none_when_all_missing():
    assert mean_score([None, None]) is None


def test_mean_score_empty_list():
    assert mean_score([]) is None


def _record(
    tags=None,
    precision=1.0,
    recall=1.0,
    citation_validity_rate=1.0,
    groundedness_score=0.9,
    correctness_score=0.5,
    refused=False,
):
    return {
        "tags": tags or [],
        "retrieval_metrics": (
            {"precision": precision, "recall": recall}
            if precision is not None
            else None
        ),
        "citation_validity_rate": citation_validity_rate,
        "groundedness": (
            {"score": groundedness_score, "unsupported_claims": []}
            if groundedness_score is not None
            else None
        ),
        "correctness": {"score": correctness_score, "fact_results": []},
        "refused": refused,
    }


def test_refusal_accuracy_none_when_no_refusal_tagged_questions():
    records = [_record(tags=["component:svm"])]
    assert refusal_accuracy(records) is None


def test_refusal_accuracy_counts_only_refusal_tagged_questions():
    records = [
        _record(tags=["refusal"], refused=True),
        _record(tags=["refusal"], refused=False),
        _record(tags=["component:svm"], refused=False),
    ]
    assert refusal_accuracy(records) == 0.5


def test_build_summary_rolls_up_across_questions():
    records = [
        _record(
            precision=1.0, recall=1.0, correctness_score=1.0, groundedness_score=1.0
        ),
        _record(
            precision=0.5, recall=0.5, correctness_score=0.0, groundedness_score=0.0
        ),
    ]

    summary = build_summary(records)

    assert summary["n_questions"] == 2
    assert summary["mean_precision"] == 0.75
    assert summary["mean_recall"] == 0.75
    assert summary["mean_correctness"] == 0.5
    assert summary["mean_groundedness"] == 0.5
    assert summary["mean_citation_validity_rate"] == 1.0
    assert summary["refusal_accuracy"] is None


def test_build_summary_skips_questions_with_no_target_issues():
    records = [
        _record(precision=None, recall=None),
        _record(precision=1.0, recall=1.0),
    ]

    summary = build_summary(records)

    assert summary["mean_precision"] == 1.0
    assert summary["mean_recall"] == 1.0


def test_build_summary_skips_refused_answers_for_groundedness():
    records = [
        _record(groundedness_score=None, refused=True, citation_validity_rate=None),
        _record(groundedness_score=0.8),
    ]

    summary = build_summary(records)

    assert summary["mean_groundedness"] == 0.8
    assert summary["mean_citation_validity_rate"] == 1.0
