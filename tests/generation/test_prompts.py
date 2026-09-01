from src.generation.prompts import build_messages


def test_build_messages_includes_question_and_chunk_metadata():
    hits = [
        {
            "issue_number": 42,
            "component": "svm",
            "state": "closed",
            "text": "some chunk text",
            "score": 0.9,
            "matched_via": "vector",
        }
    ]

    messages = build_messages("why does svm fail?", hits)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert "why does svm fail?" in user_content
    assert "42" in user_content
    assert "svm" in user_content
    assert "closed" in user_content
    assert "some chunk text" in user_content


def test_build_messages_excludes_internal_pipeline_fields():
    hits = [
        {
            "issue_number": 42,
            "component": "svm",
            "state": "closed",
            "text": "some chunk text",
            "score": 0.913,
            "matched_via": "vector+filter",
        }
    ]

    user_content = build_messages("question", hits)[1]["content"]

    assert "0.913" not in user_content
    assert "matched_via" not in user_content
    assert "vector+filter" not in user_content
