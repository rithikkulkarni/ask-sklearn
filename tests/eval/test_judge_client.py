from unittest.mock import Mock

import src.eval.judge_client as judge_client_module
from src.eval.judge_client import configure, judge_structured


def test_configure_creates_and_stores_a_client(monkeypatch):
    mock_client = Mock()
    mock_client_cls = Mock(return_value=mock_client)
    monkeypatch.setattr("src.eval.judge_client.genai.Client", mock_client_cls)

    configure("fake-api-key")

    mock_client_cls.assert_called_once_with(api_key="fake-api-key")
    assert judge_client_module._client is mock_client


def test_judge_structured_calls_gemini_and_parses_json(monkeypatch):
    mock_response = Mock()
    mock_response.text = '{"groundedness_score": 0.5, "unsupported_claims": []}'

    mock_client = Mock()
    mock_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr(judge_client_module, "_client", mock_client)

    schema = {"type": "object"}
    result = judge_structured("some prompt", schema, "gemini-3.5-flash-lite", 0.0)

    assert result == {"groundedness_score": 0.5, "unsupported_claims": []}

    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-3.5-flash-lite"
    assert kwargs["contents"] == "some prompt"
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_json_schema == schema
    assert kwargs["config"].temperature == 0.0
