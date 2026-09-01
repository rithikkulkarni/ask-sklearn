"""Keyword/regex extraction of component and version filter values from a query.

This is intentionally simple: it's a caller-supplied hint for hybrid retrieval, not a
full query-understanding step (LLM-based extraction is a future ticket). It's
imprecise when a question doesn't explicitly name a component or version -- that
tradeoff is accepted for now.
"""

import re

from src.chunking.chunk_issues import VERSION_RE

# The distinct `component` values actually present in the indexed corpus (derived
# from data/processed/chunks.jsonl at the time this was written). Frozen as a
# constant rather than read from data at runtime, so retrieval has no dependency on
# data/ being present (e.g. in a deployed environment where only Qdrant + OpenAI are
# reachable). Needs regenerating if the corpus changes significantly.
KNOWN_COMPONENTS = [
    "base",
    "calibration",
    "cluster",
    "common",
    "compose",
    "covariance",
    "cross_decomposition",
    "datasets",
    "decomposition",
    "discriminant_analysis",
    "ensemble",
    "feature_extraction",
    "feature_selection",
    "gaussian_process",
    "impute",
    "inspection",
    "isotonic",
    "kernel_approximation",
    "linear_model",
    "manifold",
    "metrics",
    "mixture",
    "model_selection",
    "multiclass",
    "multioutput",
    "naive_bayes",
    "neighbors",
    "neural_network",
    "pipeline",
    "preprocessing",
    "random_projection",
    "semi_supervised",
    "svm",
    "test-suite",
    "tree",
    "utils",
]


def _component_pattern(component: str) -> re.Pattern:
    """Build a pattern matching `component`'s words joined by _, -, or whitespace,
    so "model_selection" also matches "model selection" / "model-selection".
    """
    words = [re.escape(word) for word in re.split(r"[_\-\s]+", component)]
    return re.compile(r"\b" + r"[_\s-]".join(words) + r"\b", re.IGNORECASE)


_COMPONENT_PATTERNS = {
    component: _component_pattern(component) for component in KNOWN_COMPONENTS
}


def extract_filters(query: str) -> dict[str, list[str]]:
    """Extract component/version filter values mentioned in a query.

    Returns only the fields that had at least one match, e.g. {"component": [...]}
    or {} if nothing was found -- an empty dict signals "no filter, vector-only".
    """
    filters = {}

    matched_components = [
        component
        for component, pattern in _COMPONENT_PATTERNS.items()
        if pattern.search(query)
    ]
    if matched_components:
        filters["component"] = matched_components

    version_match = VERSION_RE.search(query)
    if version_match:
        filters["version"] = [version_match.group(1)]

    return filters
