from unittest.mock import Mock

from src.embedding.migrate_to_cloud import migrate


def _fake_point(point_id, vector, payload):
    return Mock(id=point_id, vector=vector, payload=payload)


def test_migrate_copies_all_points_across_pages():
    source_client = Mock()
    source_client.scroll.side_effect = [
        ([_fake_point("id-1", [0.1], {"issue_number": 1})], "next-offset"),
        ([_fake_point("id-2", [0.2], {"issue_number": 2})], None),
    ]
    dest_client = Mock()

    total = migrate(source_client, dest_client, "my_collection")

    assert total == 2
    assert dest_client.upsert.call_count == 2

    first_call_points = dest_client.upsert.call_args_list[0].kwargs["points"]
    assert first_call_points[0].id == "id-1"
    assert first_call_points[0].vector == [0.1]
    assert first_call_points[0].payload == {"issue_number": 1}


def test_migrate_stops_when_source_collection_is_empty():
    source_client = Mock()
    source_client.scroll.return_value = ([], None)
    dest_client = Mock()

    total = migrate(source_client, dest_client, "my_collection")

    assert total == 0
    dest_client.upsert.assert_not_called()


def test_migrate_reads_and_writes_with_correct_collection_name():
    source_client = Mock()
    source_client.scroll.return_value = (
        [_fake_point("id-1", [0.1], {"issue_number": 1})],
        None,
    )
    dest_client = Mock()

    migrate(source_client, dest_client, "my_collection")

    _, scroll_kwargs = source_client.scroll.call_args
    assert scroll_kwargs["collection_name"] == "my_collection"
    assert scroll_kwargs["with_payload"] is True
    assert scroll_kwargs["with_vectors"] is True

    _, upsert_kwargs = dest_client.upsert.call_args
    assert upsert_kwargs["collection_name"] == "my_collection"
