from unittest.mock import Mock

from src.eval.dataset import EvalQuestion
from src.eval.run_eval import run_question


def _fake_config():
    config = Mock()
    config.eval.judge_model = "gemini-3.5-flash-lite"
    config.eval.judge_temperature = 0.0
    return config


def _fake_hit(issue_number, score=0.9):
    return {
        "chunk_id": f"{issue_number}-0",
        "issue_number": issue_number,
        "component": "svm",
        "state": "open",
        "text": "chunk text",
        "score": score,
    }


def _fake_question(query="why does X happen?", target_issue_numbers=None, tags=None):
    return EvalQuestion(
        id="q01",
        query=query,
        target_issue_numbers=target_issue_numbers or [1],
        rubric_facts=["fact one"],
        tags=tags or [],
        smoke=False,
    )


def test_baseline_mode_retrieves_with_no_filters(monkeypatch):
    mock_retrieve = Mock(return_value=[_fake_hit(1)])
    monkeypatch.setattr("src.eval.run_eval.retrieve", mock_retrieve)
    monkeypatch.setattr(
        "src.eval.run_eval.generate",
        Mock(
            return_value={
                "answer": "answer",
                "citations": [1],
                "refused": False,
                "refusal_reason": None,
                "confidence_score": 0.9,
            }
        ),
    )
    monkeypatch.setattr(
        "src.eval.run_eval.judge_groundedness",
        Mock(return_value={"score": 1.0, "unsupported_claims": []}),
    )
    monkeypatch.setattr(
        "src.eval.run_eval.judge_correctness",
        Mock(return_value={"score": 1.0, "fact_results": []}),
    )

    run_question(_fake_question(), "baseline", Mock(), Mock(), _fake_config())

    args, _ = mock_retrieve.call_args
    query, filters = args[0], args[1]
    assert query == "why does X happen?"
    assert filters == {}


def test_hybrid_mode_retrieves_with_extracted_filters(monkeypatch):
    mock_retrieve = Mock(return_value=[_fake_hit(1)])
    monkeypatch.setattr("src.eval.run_eval.retrieve", mock_retrieve)
    monkeypatch.setattr(
        "src.eval.run_eval.extract_filters",
        Mock(return_value={"component": ["svm"]}),
    )
    monkeypatch.setattr(
        "src.eval.run_eval.generate",
        Mock(
            return_value={
                "answer": "answer",
                "citations": [1],
                "refused": False,
                "refusal_reason": None,
                "confidence_score": 0.9,
            }
        ),
    )
    monkeypatch.setattr(
        "src.eval.run_eval.judge_groundedness",
        Mock(return_value={"score": 1.0, "unsupported_claims": []}),
    )
    monkeypatch.setattr(
        "src.eval.run_eval.judge_correctness",
        Mock(return_value={"score": 1.0, "fact_results": []}),
    )

    run_question(_fake_question(), "hybrid", Mock(), Mock(), _fake_config())

    args, _ = mock_retrieve.call_args
    assert args[1] == {"component": ["svm"]}


def test_groundedness_skipped_when_answer_refused(monkeypatch):
    monkeypatch.setattr("src.eval.run_eval.retrieve", Mock(return_value=[]))
    monkeypatch.setattr(
        "src.eval.run_eval.generate",
        Mock(
            return_value={
                "answer": "I don't have enough information.",
                "citations": [],
                "refused": True,
                "refusal_reason": "no_hits",
                "confidence_score": None,
            }
        ),
    )
    mock_judge_groundedness = Mock()
    monkeypatch.setattr("src.eval.run_eval.judge_groundedness", mock_judge_groundedness)
    monkeypatch.setattr(
        "src.eval.run_eval.judge_correctness",
        Mock(return_value={"score": 1.0, "fact_results": []}),
    )

    record = run_question(
        _fake_question(target_issue_numbers=[], tags=["refusal"]),
        "baseline",
        Mock(),
        Mock(),
        _fake_config(),
    )

    mock_judge_groundedness.assert_not_called()
    assert record["groundedness"] is None
    assert record["refused"] is True
