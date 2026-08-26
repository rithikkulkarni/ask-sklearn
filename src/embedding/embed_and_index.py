"""Embed chunks (data/processed/chunks.jsonl) with OpenAI and index them into Qdrant.

Skips chunks whose point ID already exists in the collection, so re-running after a
partial run (or after re-chunking issues that didn't change) doesn't force a full
re-embed.

Usage:
    python -m src.embedding.embed_and_index
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.config import load_config
from src.embedding.openai_client import embed_texts
from src.embedding.qdrant_store import (
    chunk_point_id,
    ensure_collection,
    existing_ids,
    get_client,
    upsert_chunks,
)

CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def load_chunks():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (add it to .env)")

    config = load_config()
    embedding_cfg = config.embedding
    store_cfg = config.vector_store

    openai_client = OpenAI(api_key=api_key)
    qdrant_client = get_client(store_cfg.host, store_cfg.port)
    ensure_collection(
        qdrant_client,
        store_cfg.collection_name,
        store_cfg.vector_size,
        store_cfg.distance,
    )

    already_indexed = existing_ids(qdrant_client, store_cfg.collection_name)

    chunks = list(load_chunks())
    new_chunks = [
        chunk
        for chunk in chunks
        if chunk_point_id(chunk["chunk_id"]) not in already_indexed
    ]

    print(f"{len(chunks)} chunks total, {len(new_chunks)} new (not yet indexed).")

    total_indexed = 0
    batch_size = embedding_cfg.batch_size
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i : i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        vectors = embed_texts(openai_client, texts, embedding_cfg.model)
        upsert_chunks(qdrant_client, store_cfg.collection_name, batch, vectors)
        total_indexed += len(batch)
        print(f"Indexed {total_indexed}/{len(new_chunks)} new chunks...")

    print(
        f"Done. {total_indexed} chunks embedded and indexed into "
        f"'{store_cfg.collection_name}'."
    )


if __name__ == "__main__":
    main()
