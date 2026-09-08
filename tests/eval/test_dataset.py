import json

from src.eval.dataset import load_questions


def _write_questions(tmp_path, questions):
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps(questions), encoding="utf-8")
    return path


def test_load_questions_parses_all_fields(tmp_path):
    path = _write_questions(
        tmp_path,
        [
            {
                "id": "q01",
                "query": "why does X happen?",
                "target_issue_numbers": [42],
                "rubric_facts": ["fact one", "fact two"],
                "tags": ["component:svm"],
                "smoke": True,
            }
        ],
    )

    questions = load_questions(path)

    assert len(questions) == 1
    q = questions[0]
    assert q.id == "q01"
    assert q.query == "why does X happen?"
    assert q.target_issue_numbers == [42]
    assert q.rubric_facts == ["fact one", "fact two"]
    assert q.tags == ["component:svm"]
    assert q.smoke is True


def test_load_questions_defaults_missing_tags_to_empty_list(tmp_path):
    path = _write_questions(
        tmp_path,
        [
            {
                "id": "q01",
                "query": "q",
                "target_issue_numbers": [],
                "rubric_facts": [],
                "smoke": False,
            }
        ],
    )

    questions = load_questions(path)

    assert questions[0].tags == []


def test_smoke_only_filters_to_smoke_questions(tmp_path):
    path = _write_questions(
        tmp_path,
        [
            {
                "id": "q01",
                "query": "q1",
                "target_issue_numbers": [1],
                "rubric_facts": [],
                "tags": [],
                "smoke": True,
            },
            {
                "id": "q02",
                "query": "q2",
                "target_issue_numbers": [2],
                "rubric_facts": [],
                "tags": [],
                "smoke": False,
            },
        ],
    )

    questions = load_questions(path, smoke_only=True)

    assert [q.id for q in questions] == ["q01"]


def test_the_real_test_set_loads_and_has_expected_smoke_count():
    from src.eval.dataset import DEFAULT_PATH

    questions = load_questions(DEFAULT_PATH)
    smoke_questions = [q for q in questions if q.smoke]

    assert 30 <= len(questions) <= 50
    assert 5 <= len(smoke_questions) <= 8
    assert len({q.id for q in questions}) == len(questions)
