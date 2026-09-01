from unittest.mock import Mock

from src.generation.generate import generate


def _fake_config(min_top_score=0.75):
    config = Mock()
    config.generation.model = "gpt-4o-mini"
    config.generation.temperature = 0.1
    config.generation.min_top_score = min_top_score
    return config


def _fake_hit(issue_number, score, **overrides):
    hit = {
        "chunk_id": f"{issue_number}-0",
        "issue_number": issue_number,
        "issue_url": f"https://github.com/scikit-learn/scikit-learn/issues/{issue_number}",
        "component": "svm",
        "version": "1.3.2",
        "state": "open",
        "text": "chunk text",
        "score": score,
        "matched_via": "vector",
    }
    hit.update(overrides)
    return hit


def test_refuses_on_empty_hits_without_calling_llm(monkeypatch):
    mock_generate_structured = Mock()
    monkeypatch.setattr(
        "src.generation.generate.generate_structured", mock_generate_structured
    )

    result = generate("question", [], Mock(), _fake_config())

    assert result["refused"] is True
    assert result["refusal_reason"] == "no_hits"
    assert result["confidence_score"] is None
    assert result["citations"] == []
    mock_generate_structured.assert_not_called()


def test_refuses_on_low_top_score_without_calling_llm(monkeypatch):
    mock_generate_structured = Mock()
    monkeypatch.setattr(
        "src.generation.generate.generate_structured", mock_generate_structured
    )

    hits = [_fake_hit(1, 0.5)]
    result = generate("question", hits, Mock(), _fake_config(min_top_score=0.75))

    assert result["refused"] is True
    assert result["refusal_reason"] == "low_confidence"
    assert result["confidence_score"] == 0.5
    mock_generate_structured.assert_not_called()


def test_model_refusal_path_uses_models_own_answer(monkeypatch):
    monkeypatch.setattr(
        "src.generation.generate.generate_structured",
        Mock(
            return_value={
                "answer": "I can't find support for this.",
                "citations": [],
                "refused": True,
            }
        ),
    )

    hits = [_fake_hit(1, 0.9)]
    result = generate("question", hits, Mock(), _fake_config())

    assert result["refused"] is True
    assert result["refusal_reason"] == "not_grounded"
    assert result["answer"] == "I can't find support for this."
    assert result["citations"] == []
    assert result["confidence_score"] == 0.9


def test_filters_out_hallucinated_citations(monkeypatch):
    monkeypatch.setattr(
        "src.generation.generate.generate_structured",
        Mock(
            return_value={
                "answer": "answer text",
                "citations": [1, 9999],
                "refused": False,
            }
        ),
    )

    hits = [_fake_hit(1, 0.9)]
    result = generate("question", hits, Mock(), _fake_config())

    assert result["citations"] == [1]


def test_successful_output_shape(monkeypatch):
    monkeypatch.setattr(
        "src.generation.generate.generate_structured",
        Mock(
            return_value={
                "answer": "answer text",
                "citations": [1, 2],
                "refused": False,
            }
        ),
    )

    hits = [_fake_hit(1, 0.9), _fake_hit(2, 0.8)]
    result = generate("question", hits, Mock(), _fake_config())

    assert result == {
        "answer": "answer text",
        "citations": [1, 2],
        "refused": False,
        "refusal_reason": None,
        "confidence_score": 0.9,
    }
