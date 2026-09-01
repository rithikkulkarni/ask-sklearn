from src.retrieval.filter_extraction import extract_filters


def test_extract_filters_finds_single_component():
    filters = extract_filters("Why does GridSearchCV in model_selection fail?")
    assert filters == {"component": ["model_selection"]}


def test_extract_filters_finds_multiple_components():
    filters = extract_filters("Is this a bug in both ensemble and svm?")
    assert filters == {"component": ["ensemble", "svm"]}


def test_extract_filters_matches_space_and_hyphen_variants():
    filters = extract_filters("the kernel approximation code looks off")
    assert filters == {"component": ["kernel_approximation"]}


def test_extract_filters_finds_version():
    filters = extract_filters(
        "This started happening after upgrading to scikit-learn 1.3.2"
    )
    assert filters == {"version": ["1.3.2"]}


def test_extract_filters_finds_component_and_version():
    filters = extract_filters("ensemble broke on sklearn 1.4.0")
    assert filters == {"component": ["ensemble"], "version": ["1.4.0"]}


def test_extract_filters_returns_empty_dict_when_nothing_matches():
    assert extract_filters("Why is my model slow?") == {}


def test_extract_filters_avoids_substring_false_positive():
    assert extract_filters("what are svms good for?") == {}
