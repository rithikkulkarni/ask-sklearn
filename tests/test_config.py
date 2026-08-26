import yaml

from src.config import load_config


def test_load_config_parses_all_sections(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "chunking": {
                    "chunk_size_tokens": 400,
                    "chunk_overlap_tokens": 50,
                    "tokenizer_encoding": "cl100k_base",
                    "module_label_prefix": "module:",
                },
                "embedding": {
                    "model": "text-embedding-3-small",
                    "batch_size": 100,
                },
                "vector_store": {
                    "collection_name": "sklearn_issue_chunks",
                    "vector_size": 1536,
                    "distance": "Cosine",
                    "host": "localhost",
                    "port": 6333,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.chunking.chunk_size_tokens == 400
    assert config.chunking.module_label_prefix == "module:"
    assert config.embedding.model == "text-embedding-3-small"
    assert config.embedding.batch_size == 100
    assert config.vector_store.collection_name == "sklearn_issue_chunks"
    assert config.vector_store.vector_size == 1536
    assert config.vector_store.distance == "Cosine"
    assert config.vector_store.host == "localhost"
    assert config.vector_store.port == 6333
