"""One-off migration: copy every point (vector + payload) from the local Qdrant
instance straight into a Qdrant Cloud cluster.

This avoids two more expensive/fragile alternatives: re-embedding everything via
OpenAI again (costs money, and isn't necessary since the vectors already exist),
or relying on Qdrant's snapshot export/import tooling (extra file plumbing). It
just scrolls every point out of the local collection and upserts it into the
cloud one, in batches.

Requires the local Qdrant instance (Docker) to be running, since it's read from
directly during the migration.

Usage:
    python -m src.embedding.migrate_to_cloud
"""

import os

from dotenv import load_dotenv
from qdrant_client.models import PointStruct

from src.config import load_config
from src.embedding.qdrant_store import ensure_collection, get_client

BATCH_SIZE = 500


def migrate(source_client, dest_client, collection_name: str) -> int:
    total = 0
    offset = None
    while True:
        points, offset = source_client.scroll(
            collection_name=collection_name,
            limit=BATCH_SIZE,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            break

        dest_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points
            ],
        )
        total += len(points)
        print(f"Migrated {total} points so far...")

        if offset is None:
            break

    return total


def main():
    load_dotenv()
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise RuntimeError(
            "QDRANT_URL and QDRANT_API_KEY must both be set (add them to .env)"
        )

    config = load_config()
    store_cfg = config.vector_store

    source_client = get_client(store_cfg.host, store_cfg.port)
    dest_client = get_client(
        store_cfg.host, store_cfg.port, url=qdrant_url, api_key=qdrant_api_key
    )

    ensure_collection(
        dest_client,
        store_cfg.collection_name,
        store_cfg.vector_size,
        store_cfg.distance,
    )

    total = migrate(source_client, dest_client, store_cfg.collection_name)
    print(f"Done. Migrated {total} points to Qdrant Cloud.")


if __name__ == "__main__":
    main()
