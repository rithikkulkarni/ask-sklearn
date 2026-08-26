from unittest.mock import Mock

from src.embedding.openai_client import embed_texts


def test_embed_texts_calls_create_with_model_and_input_and_preserves_order():
    fake_response = Mock()
    fake_response.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
    client = Mock()
    client.embeddings.create.return_value = fake_response

    vectors = embed_texts(client, ["first", "second"], model="text-embedding-3-small")

    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small", input=["first", "second"]
    )
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
