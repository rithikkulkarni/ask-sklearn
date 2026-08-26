import json
from unittest.mock import Mock

from src.embedding.embed_and_index import load_chunks, main
from src.embedding.qdrant_store import chunk_point_id


def test_load_chunks_reads_jsonl(tmp_path, monkeypatch):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps({"chunk_id": cid}) for cid in ["1-0", "1-1"]) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("src.embedding.embed_and_index.CHUNKS_PATH", chunks_path)

    chunks = list(load_chunks())

    assert [chunk["chunk_id"] for chunk in chunks] == ["1-0", "1-1"]


def test_main_skips_already_indexed_and_batches_new_chunks(tmp_path, monkeypatch):
    records = [
        {
            "chunk_id": "1-0",
            "issue_number": 1,
            "component": "a",
            "version": None,
            "text": "t0",
        },
        {
            "chunk_id": "1-1",
            "issue_number": 1,
            "component": "a",
            "version": None,
            "text": "t1",
        },
        {
            "chunk_id": "1-2",
            "issue_number": 1,
            "component": "a",
            "version": None,
            "text": "t2",
        },
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr("src.embedding.embed_and_index.CHUNKS_PATH", chunks_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.embedding.embed_and_index.load_dotenv", Mock())
    monkeypatch.setattr(
        "src.embedding.embed_and_index.OpenAI", Mock(return_value=Mock())
    )

    fake_config = Mock()
    fake_config.embedding.model = "text-embedding-3-small"
    fake_config.embedding.batch_size = 2
    fake_config.vector_store.collection_name = "col"
    fake_config.vector_store.vector_size = 1536
    fake_config.vector_store.distance = "Cosine"
    fake_config.vector_store.host = "localhost"
    fake_config.vector_store.port = 6333
    monkeypatch.setattr(
        "src.embedding.embed_and_index.load_config", Mock(return_value=fake_config)
    )

    fake_qdrant_client = Mock()
    monkeypatch.setattr(
        "src.embedding.embed_and_index.get_client",
        Mock(return_value=fake_qdrant_client),
    )
    mock_ensure_collection = Mock()
    monkeypatch.setattr(
        "src.embedding.embed_and_index.ensure_collection", mock_ensure_collection
    )

    already_indexed = {chunk_point_id("1-0")}
    monkeypatch.setattr(
        "src.embedding.embed_and_index.existing_ids", Mock(return_value=already_indexed)
    )

    mock_embed_texts = Mock(
        side_effect=lambda client, texts, model: [[0.0] for _ in texts]
    )
    monkeypatch.setattr("src.embedding.embed_and_index.embed_texts", mock_embed_texts)

    mock_upsert_chunks = Mock()
    monkeypatch.setattr(
        "src.embedding.embed_and_index.upsert_chunks", mock_upsert_chunks
    )

    main()

    mock_ensure_collection.assert_called_once_with(
        fake_qdrant_client, "col", 1536, "Cosine"
    )

    mock_embed_texts.assert_called_once()
    assert mock_embed_texts.call_args.args[1] == ["t1", "t2"]
    assert mock_embed_texts.call_args.args[2] == "text-embedding-3-small"

    upserted_chunk_ids = [
        record["chunk_id"]
        for call in mock_upsert_chunks.call_args_list
        for record in call.args[2]
    ]
    assert upserted_chunk_ids == ["1-1", "1-2"]
