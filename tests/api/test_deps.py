from unittest.mock import Mock

import pytest

from src.api.deps import build_app_state, get_app_state


def test_build_app_state_raises_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_app_state()


def test_build_app_state_wires_clients(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    mock_config = Mock()
    monkeypatch.setattr("src.api.deps.load_config", Mock(return_value=mock_config))

    mock_openai_cls = Mock()
    monkeypatch.setattr("src.api.deps.OpenAI", mock_openai_cls)

    mock_get_client = Mock()
    monkeypatch.setattr("src.api.deps.get_client", mock_get_client)

    state = build_app_state()

    assert state.config is mock_config
    mock_openai_cls.assert_called_once_with(api_key="fake-key")
    mock_get_client.assert_called_once_with(
        mock_config.vector_store.host,
        mock_config.vector_store.port,
        url=None,
        api_key=None,
    )


def test_build_app_state_passes_through_qdrant_cloud_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("QDRANT_URL", "https://cluster.cloud.qdrant.io:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-key")

    monkeypatch.setattr("src.api.deps.load_config", Mock(return_value=Mock()))
    monkeypatch.setattr("src.api.deps.OpenAI", Mock())
    mock_get_client = Mock()
    monkeypatch.setattr("src.api.deps.get_client", mock_get_client)

    build_app_state()

    _, kwargs = mock_get_client.call_args
    assert kwargs["url"] == "https://cluster.cloud.qdrant.io:6333"
    assert kwargs["api_key"] == "qdrant-key"


def test_get_app_state_reads_from_request_app_state():
    request = Mock()
    request.app.state.app_state = "the-state"

    assert get_app_state(request) == "the-state"
