from src.eval.retrieval_metrics import precision_recall


def test_returns_none_when_no_target_issues():
    assert precision_recall([1, 2, 3], []) is None


def test_perfect_precision_and_recall():
    result = precision_recall([1, 2], [1, 2])
    assert result == {"precision": 1.0, "recall": 1.0}


def test_partial_overlap():
    # 2 of 3 retrieved are relevant; 2 of 4 target issues were found.
    result = precision_recall([1, 2, 3], [1, 2, 5, 6])
    assert result == {"precision": 2 / 3, "recall": 2 / 4}


def test_no_overlap():
    result = precision_recall([1, 2], [3, 4])
    assert result == {"precision": 0.0, "recall": 0.0}


def test_empty_retrieval_with_target_issues():
    result = precision_recall([], [1, 2])
    assert result == {"precision": 0.0, "recall": 0.0}


def test_duplicate_retrieved_issue_numbers_are_deduped():
    # Multiple chunks from the same issue shouldn't inflate precision's denominator.
    result = precision_recall([1, 1, 1], [1])
    assert result == {"precision": 1.0, "recall": 1.0}
