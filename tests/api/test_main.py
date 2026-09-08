from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.api.deps import AppState
from src.api.rate_limit import limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def client(monkeypatch):
    fake_state = AppState(config=Mock(), openai_client=Mock(), qdrant_client=Mock())
    monkeypatch.setattr("src.api.main.build_app_state", lambda: fake_state)

    from src.api.main import app

    # raise_server_exceptions=False: Starlette's ServerErrorMiddleware always
    # re-raises after calling a registered Exception handler (so servers can
    # log it) -- TestClient surfaces that re-raise by default, which would
    # break the test even though real clients (uvicorn) still get the clean
    # response our handler sent. This makes the client behave like a real one.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _fake_hit(issue_number, score=0.9):
    return {
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


def _mock_pipeline(monkeypatch, hits=None, generate_result=None):
    monkeypatch.setattr("src.api.main.retrieve", Mock(return_value=hits or []))
    monkeypatch.setattr(
        "src.api.main.generate",
        Mock(
            return_value=generate_result
            or {
                "answer": "answer text",
                "citations": [1],
                "refused": False,
                "refusal_reason": None,
                "confidence_score": 0.9,
            }
        ),
    )


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_happy_path_returns_lean_response(client, monkeypatch):
    _mock_pipeline(
        monkeypatch,
        hits=[_fake_hit(1)],
        generate_result={
            "answer": "SGDRegressor needs a 1D y.",
            "citations": [1],
            "refused": False,
            "refusal_reason": None,
            "confidence_score": 0.9,
        },
    )

    response = client.post("/query", json={"question": "why does X fail?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "SGDRegressor needs a 1D y."
    assert body["citations"] == [
        {
            "issue_number": 1,
            "url": "https://github.com/scikit-learn/scikit-learn/issues/1",
        }
    ]
    assert body["refused"] is False
    assert body["refusal_reason"] is None
    # response must stay lean -- no raw scores or chunk text leak into the API
    assert "confidence_score" not in body
    assert "score" not in str(body)


def test_query_refusal_path(client, monkeypatch):
    _mock_pipeline(
        monkeypatch,
        hits=[],
        generate_result={
            "answer": "I don't have enough information.",
            "citations": [],
            "refused": True,
            "refusal_reason": "no_hits",
            "confidence_score": None,
        },
    )

    response = client.post("/query", json={"question": "unrelated question"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["refusal_reason"] == "no_hits"
    assert body["citations"] == []


def test_query_defaults_to_hybrid_mode(client, monkeypatch):
    mock_retrieve = Mock(return_value=[])
    monkeypatch.setattr("src.api.main.retrieve", mock_retrieve)
    monkeypatch.setattr(
        "src.api.main.extract_filters", Mock(return_value={"component": ["svm"]})
    )
    monkeypatch.setattr(
        "src.api.main.generate",
        Mock(
            return_value={
                "answer": "a",
                "citations": [],
                "refused": False,
                "refusal_reason": None,
                "confidence_score": 0.9,
            }
        ),
    )

    client.post("/query", json={"question": "svm question"})

    args, _ = mock_retrieve.call_args
    assert args[1] == {"component": ["svm"]}


def test_query_baseline_mode_uses_no_filters(client, monkeypatch):
    mock_retrieve = Mock(return_value=[])
    monkeypatch.setattr("src.api.main.retrieve", mock_retrieve)
    monkeypatch.setattr(
        "src.api.main.generate",
        Mock(
            return_value={
                "answer": "a",
                "citations": [],
                "refused": False,
                "refusal_reason": None,
                "confidence_score": 0.9,
            }
        ),
    )

    client.post(
        "/query", json={"question": "svm question", "retrieval_mode": "baseline"}
    )

    args, _ = mock_retrieve.call_args
    assert args[1] == {}


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422


def test_query_rejects_missing_question(client):
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_rejects_invalid_retrieval_mode(client):
    response = client.post(
        "/query", json={"question": "q", "retrieval_mode": "not-a-real-mode"}
    )

    assert response.status_code == 422


def test_unhandled_exception_returns_clean_502(client, monkeypatch):
    monkeypatch.setattr(
        "src.api.main.retrieve", Mock(side_effect=ConnectionError("qdrant down"))
    )

    response = client.post("/query", json={"question": "q"})

    assert response.status_code == 502
    assert "qdrant down" not in response.text
    assert response.json() == {"detail": "Upstream service error. Please try again."}


def test_rate_limit_enforced_per_config(client, monkeypatch):
    # config.yaml's api.rate_limit is "20/minute" -- exercise the real configured
    # value rather than a hardcoded number, so this test catches drift if that
    # value ever changes.
    from src.api.main import _config

    limit = int(_config.api.rate_limit.split("/")[0])

    _mock_pipeline(monkeypatch)

    for _ in range(limit):
        response = client.post("/query", json={"question": "q"})
        assert response.status_code == 200

    response = client.post("/query", json={"question": "q"})
    assert response.status_code == 429
