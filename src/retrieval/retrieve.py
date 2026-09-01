"""Hybrid retrieval: vector similarity search over the Qdrant index, optionally
narrowed by component/version filters.

This function takes `filters` as an explicit parameter rather than deciding them
itself -- it doesn't know or care how they were determined. For now, the caller is
expected to produce them via filter_extraction.extract_filters(query) (regex-based);
a future ticket can swap in LLM-based extraction at the call site without touching
this function at all.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from src.config import Config
from src.embedding.openai_client import embed_texts

ISSUE_URL_TEMPLATE = (
    "https://github.com/scikit-learn/scikit-learn/issues/{issue_number}"
)


def _build_filter(filters: dict[str, list[str]]) -> Filter | None:
    if not filters:
        return None
    return Filter(
        must=[
            FieldCondition(key=field, match=MatchAny(any=values))
            for field, values in filters.items()
        ]
    )


def retrieve(
    query: str,
    filters: dict[str, list[str]],
    qdrant_client: QdrantClient,
    openai_client,
    config: Config,
) -> list[dict]:
    retrieval_cfg = config.retrieval

    allowed_filters = {
        field: values
        for field, values in filters.items()
        if field in retrieval_cfg.filter_fields
    }
    query_filter = _build_filter(allowed_filters)

    (query_vector,) = embed_texts(openai_client, [query], config.embedding.model)

    response = qdrant_client.query_points(
        collection_name=config.vector_store.collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=retrieval_cfg.overfetch_k,
        with_payload=True,
    )
    candidates = response.points

    above_threshold = [
        point for point in candidates if point.score >= retrieval_cfg.score_threshold
    ]
    if len(above_threshold) >= retrieval_cfg.min_k:
        kept = above_threshold[: retrieval_cfg.max_k]
    else:
        # Nothing (or too little) cleared the threshold -- backfill with the
        # next-highest-scoring candidates regardless of threshold, up to min_k.
        kept = candidates[: retrieval_cfg.min_k]

    matched_via = "vector+filter" if query_filter is not None else "vector"

    return [
        {
            "chunk_id": point.payload["chunk_id"],
            "issue_number": point.payload["issue_number"],
            "issue_url": ISSUE_URL_TEMPLATE.format(
                issue_number=point.payload["issue_number"]
            ),
            "component": point.payload["component"],
            "version": point.payload["version"],
            "state": point.payload["state"],
            "text": point.payload["text"],
            "score": point.score,
            "matched_via": matched_via,
        }
        for point in kept
    ]
