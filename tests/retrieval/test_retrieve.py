from unittest.mock import Mock

from src.retrieval.retrieve import retrieve


def _fake_point(chunk_id, score, **overrides):
    payload = {
        "chunk_id": chunk_id,
        "issue_number": 1,
        "component": "ensemble",
        "version": "1.3.2",
        "state": "open",
        "text": f"text for {chunk_id}",
    }
    payload.update(overrides)
    point = Mock()
    point.score = score
    point.payload = payload
    return point


def _fake_config():
    config = Mock()
    config.retrieval.overfetch_k = 30
    config.retrieval.score_threshold = 0.75
    config.retrieval.min_k = 1
    config.retrieval.max_k = 3
    config.retrieval.filter_fields = ["component", "version"]
    config.vector_store.collection_name = "col"
    config.embedding.model = "text-embedding-3-small"
    return config


def _setup(monkeypatch, points):
    qdrant_client = Mock()
    qdrant_client.query_points.return_value = Mock(points=points)
    openai_client = Mock()

    monkeypatch.setattr(
        "src.retrieval.retrieve.embed_texts", Mock(return_value=[[0.1, 0.2]])
    )
    return qdrant_client, openai_client


def test_filter_present_and_threshold_met(monkeypatch):
    points = [
        _fake_point("1-0", 0.90),
        _fake_point("1-1", 0.80),
        _fake_point("1-2", 0.60),
    ]
    qdrant_client, openai_client = _setup(monkeypatch, points)
    config = _fake_config()

    hits = retrieve(
        "question", {"component": ["ensemble"]}, qdrant_client, openai_client, config
    )

    assert [hit["chunk_id"] for hit in hits] == ["1-0", "1-1"]
    assert all(hit["matched_via"] == "vector+filter" for hit in hits)
    _, kwargs = qdrant_client.query_points.call_args
    assert kwargs["query_filter"] is not None


def test_filter_present_nothing_clears_threshold_backfills_to_min_k(monkeypatch):
    points = [
        _fake_point("1-0", 0.50),
        _fake_point("1-1", 0.40),
        _fake_point("1-2", 0.30),
    ]
    qdrant_client, openai_client = _setup(monkeypatch, points)
    config = _fake_config()
    config.retrieval.min_k = 1

    hits = retrieve(
        "question", {"component": ["ensemble"]}, qdrant_client, openai_client, config
    )

    assert [hit["chunk_id"] for hit in hits] == ["1-0"]


def test_no_filter_extracted_falls_back_to_vector_only(monkeypatch):
    points = [_fake_point("1-0", 0.90)]
    qdrant_client, openai_client = _setup(monkeypatch, points)
    config = _fake_config()

    hits = retrieve("question", {}, qdrant_client, openai_client, config)

    assert hits[0]["matched_via"] == "vector"
    _, kwargs = qdrant_client.query_points.call_args
    assert kwargs["query_filter"] is None


def test_threshold_caps_at_max_k(monkeypatch):
    points = [
        _fake_point("1-0", 0.95),
        _fake_point("1-1", 0.90),
        _fake_point("1-2", 0.85),
        _fake_point("1-3", 0.80),
        _fake_point("1-4", 0.76),
    ]
    qdrant_client, openai_client = _setup(monkeypatch, points)
    config = _fake_config()
    config.retrieval.max_k = 3

    hits = retrieve("question", {}, qdrant_client, openai_client, config)

    assert len(hits) == 3
    assert [hit["chunk_id"] for hit in hits] == ["1-0", "1-1", "1-2"]


def test_output_contract_shape(monkeypatch):
    points = [
        _fake_point(
            "1-0",
            0.9,
            issue_number=42,
            component="svm",
            version="1.2.0",
            state="closed",
            text="hi",
        )
    ]
    qdrant_client, openai_client = _setup(monkeypatch, points)
    config = _fake_config()

    hits = retrieve("question", {}, qdrant_client, openai_client, config)

    assert hits[0] == {
        "chunk_id": "1-0",
        "issue_number": 42,
        "issue_url": "https://github.com/scikit-learn/scikit-learn/issues/42",
        "component": "svm",
        "version": "1.2.0",
        "state": "closed",
        "text": "hi",
        "score": 0.9,
        "matched_via": "vector",
    }


def test_filter_fields_config_restricts_which_filters_apply(monkeypatch):
    points = [_fake_point("1-0", 0.9)]
    qdrant_client, openai_client = _setup(monkeypatch, points)
    config = _fake_config()
    config.retrieval.filter_fields = ["component"]

    retrieve(
        "question",
        {"component": ["ensemble"], "version": ["1.3.2"]},
        qdrant_client,
        openai_client,
        config,
    )

    _, kwargs = qdrant_client.query_points.call_args
    filtered_keys = [condition.key for condition in kwargs["query_filter"].must]
    assert filtered_keys == ["component"]
