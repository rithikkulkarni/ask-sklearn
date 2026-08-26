"""Thin wrapper around a single OpenAI embeddings API call.

Rate-limit retries are handled by the OpenAI SDK itself (it retries 429s with backoff
by default), unlike github_client.py's manual backoff loop, which was needed there
because that client talks to the GitHub REST API directly via `requests`.
"""

from openai import OpenAI


def embed_texts(client: OpenAI, texts: list[str], model: str) -> list[list[float]]:
    """Embed a list of texts in a single API call, preserving input order."""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]
