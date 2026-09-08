"""Thin wrapper around a single Gemini call with structured (JSON-schema) output.

Used only by the eval harness's LLM-as-judge (judge.py). Deliberately a separate
client/module from generation/llm_client.py -- the judge must be a different model
than the generator (gpt-4o-mini) to avoid self-preference bias, so this wraps a
different SDK (google-genai) entirely.
"""

import json

from google import genai
from google.genai import types

_client: genai.Client | None = None


def configure(api_key: str) -> None:
    global _client
    _client = genai.Client(api_key=api_key)


def judge_structured(
    prompt: str,
    schema: dict,
    model: str,
    temperature: float,
) -> dict:
    """Call Gemini with a JSON-schema response format and return the parsed
    JSON object.
    """
    response = _client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=temperature,
        ),
    )
    return json.loads(response.text)
