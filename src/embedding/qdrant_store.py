"""Thin wrapper around Qdrant: collection setup, existing-ID lookups, and upserts.

Point IDs are derived from `chunk_id` via uuid5, since Qdrant only accepts unsigned
integers or UUIDs as point IDs -- values like "12345-0" are neither. The original
`chunk_id` is kept in the payload so it's still readable back out.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def get_client(host: str, port: int) -> QdrantClient:
    return QdrantClient(host=host, port=port)


def ensure_collection(
    client: QdrantClient, collection_name: str, vector_size: int, distance: str
) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance(distance)),
    )


def existing_ids(client: QdrantClient, collection_name: str) -> set[str]:
    """All point IDs currently stored in the collection, via scroll pagination."""
    ids = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(point.id) for point in points)
        if offset is None:
            break
    return ids


def upsert_chunks(
    client: QdrantClient,
    collection_name: str,
    records: list[dict],
    vectors: list[list[float]],
) -> None:
    """records: chunk dicts (with chunk_id, issue_number, component, version, state, text).
    vectors: embeddings, same order/length as records.
    """
    points = [
        PointStruct(
            id=chunk_point_id(record["chunk_id"]),
            vector=vector,
            payload={
                "chunk_id": record["chunk_id"],
                "issue_number": record["issue_number"],
                "component": record["component"],
                "version": record["version"],
                "state": record["state"],
                "text": record["text"],
            },
        )
        for record, vector in zip(records, vectors)
    ]
    client.upsert(collection_name=collection_name, points=points)
