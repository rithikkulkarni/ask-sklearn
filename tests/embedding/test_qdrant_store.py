from unittest.mock import Mock

from qdrant_client.models import Distance

from src.embedding.qdrant_store import (
    chunk_point_id,
    ensure_collection,
    existing_ids,
    upsert_chunks,
)


def test_chunk_point_id_is_deterministic():
    assert chunk_point_id("123-0") == chunk_point_id("123-0")


def test_chunk_point_id_differs_for_different_chunk_ids():
    assert chunk_point_id("123-0") != chunk_point_id("123-1")


def test_ensure_collection_skips_creation_if_exists():
    client = Mock()
    client.collection_exists.return_value = True

    ensure_collection(client, "my_collection", vector_size=1536, distance="Cosine")

    client.create_collection.assert_not_called()


def test_ensure_collection_creates_when_missing():
    client = Mock()
    client.collection_exists.return_value = False

    ensure_collection(client, "my_collection", vector_size=1536, distance="Cosine")

    client.create_collection.assert_called_once()
    _, kwargs = client.create_collection.call_args
    assert kwargs["collection_name"] == "my_collection"
    assert kwargs["vectors_config"].size == 1536
    assert kwargs["vectors_config"].distance == Distance("Cosine")


def test_existing_ids_follows_scroll_pagination():
    point1 = Mock(id="id-1")
    point2 = Mock(id="id-2")
    client = Mock()
    client.scroll.side_effect = [
        ([point1], "next-offset"),
        ([point2], None),
    ]

    ids = existing_ids(client, "my_collection")

    assert ids == {"id-1", "id-2"}
    assert client.scroll.call_count == 2


def test_upsert_chunks_builds_points_with_derived_ids_and_payload():
    client = Mock()
    records = [
        {
            "chunk_id": "1-0",
            "issue_number": 1,
            "component": "ensemble",
            "version": "1.3.2",
        }
    ]
    vectors = [[0.1, 0.2]]

    upsert_chunks(client, "my_collection", records, vectors)

    client.upsert.assert_called_once()
    _, kwargs = client.upsert.call_args
    assert kwargs["collection_name"] == "my_collection"
    points = kwargs["points"]
    assert len(points) == 1
    assert points[0].id == chunk_point_id("1-0")
    assert points[0].vector == [0.1, 0.2]
    assert points[0].payload == {
        "chunk_id": "1-0",
        "issue_number": 1,
        "component": "ensemble",
        "version": "1.3.2",
    }
