"""Thin wrapper around a single OpenAI chat completion call with structured
(JSON-schema) output.

Rate-limit retries are handled by the OpenAI SDK itself, same as embed_texts in
embedding/openai_client.py.
"""

import json

from openai import OpenAI


def generate_structured(
    client: OpenAI,
    messages: list[dict],
    schema: dict,
    model: str,
    temperature: float,
) -> dict:
    """Call the chat completions API with a JSON-schema response format and
    return the parsed JSON object.
    """
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
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
    return json.loads(response.choices[0].message.content)
