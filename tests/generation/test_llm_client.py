import json
from unittest.mock import Mock

from src.generation.llm_client import generate_structured


def test_generate_structured_calls_api_and_parses_json():
    client = Mock()
    response = Mock()
    response.choices = [
        Mock(
            message=Mock(
                content=json.dumps({"answer": "hi", "citations": [1], "refused": False})
            )
        )
    ]
    client.chat.completions.create.return_value = response

    schema = {"type": "object"}
    messages = [{"role": "user", "content": "q"}]

    result = generate_structured(client, messages, schema, "gpt-4o-mini", 0.1)

    assert result == {"answer": "hi", "citations": [1], "refused": False}
    client.chat.completions.create.assert_called_once_with(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "generation_response",
                "schema": schema,
                "strict": True,
            },
        },
    )
