from unittest.mock import Mock

from src.eval.judge import citation_validity_rate, judge_correctness, judge_groundedness


def _fake_hit(issue_number, **overrides):
    hit = {
        "chunk_id": f"{issue_number}-0",
        "issue_number": issue_number,
        "text": "chunk text",
    }
    hit.update(overrides)
    return hit


def test_judge_groundedness_parses_score_and_claims(monkeypatch):
    monkeypatch.setattr(
        "src.eval.judge.judge_structured",
        Mock(
            return_value={
                "groundedness_score": 0.8,
                "unsupported_claims": ["claim not in any chunk"],
            }
        ),
    )

    result = judge_groundedness(
        "some answer", [_fake_hit(1)], "gemini-3.5-flash-lite", 0.0
    )

    assert result == {
        "score": 0.8,
        "unsupported_claims": ["claim not in any chunk"],
    }


def test_judge_correctness_computes_fraction_of_supported_facts(monkeypatch):
    monkeypatch.setattr(
        "src.eval.judge.judge_structured",
        Mock(
            return_value={
                "fact_results": [
                    {"fact": "fact one", "supported": True},
                    {"fact": "fact two", "supported": False},
                    {"fact": "fact three", "supported": True},
                ]
            }
        ),
    )

    result = judge_correctness(
        "question",
        "answer",
        ["fact one", "fact two", "fact three"],
        "gemini-3.5-flash-lite",
        0.0,
    )

    assert result["score"] == 2 / 3
    assert len(result["fact_results"]) == 3


def test_judge_correctness_handles_empty_rubric_without_dividing_by_zero(monkeypatch):
    monkeypatch.setattr(
        "src.eval.judge.judge_structured", Mock(return_value={"fact_results": []})
    )

    result = judge_correctness("q", "a", [], "gemini-3.5-flash-lite", 0.0)

    assert result["score"] == 0.0
    assert result["fact_results"] == []


def test_citation_validity_rate_all_valid():
    hits = [_fake_hit(1), _fake_hit(2)]
    assert citation_validity_rate([1, 2], hits) == 1.0


def test_citation_validity_rate_partial():
    hits = [_fake_hit(1)]
    assert citation_validity_rate([1, 999], hits) == 0.5


def test_citation_validity_rate_none_when_no_citations():
    assert citation_validity_rate([], [_fake_hit(1)]) is None
