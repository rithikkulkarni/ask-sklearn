"""Startup-time client construction and FastAPI dependency wiring.

The OpenAI and Qdrant clients are built once when the app starts, not per
request -- same reasoning as the other pipeline entrypoints (embed_and_index.py,
run_eval.py): they're safe to share across calls and shouldn't be rebuilt every
time.
"""

import os
from dataclasses import dataclass

from fastapi import Request
from openai import OpenAI
from qdrant_client import QdrantClient

from src.config import Config, load_config
from src.embedding.qdrant_store import get_client


@dataclass
class AppState:
    config: Config
    openai_client: OpenAI
    qdrant_client: QdrantClient


def build_app_state() -> AppState:
    config = load_config()

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set (add it to .env)")

    openai_client = OpenAI(api_key=openai_key)
    qdrant_client = get_client(
        config.vector_store.host,
        config.vector_store.port,
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )

    return AppState(
        config=config, openai_client=openai_client, qdrant_client=qdrant_client
    )


def get_app_state(request: Request) -> AppState:
    return request.app.state.app_state
