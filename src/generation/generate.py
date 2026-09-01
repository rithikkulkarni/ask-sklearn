"""Generation: turn a question + retrieved hits into a grounded, cited answer.

Takes `hits` as an explicit input rather than calling retrieve() internally --
mirrors the same principle applied to retrieve()/filters. Generation shouldn't know
or care which retrieval mode produced its input, so it can stay the fixed variable
when Milestone 6 compares baseline vs. hybrid retrieval.

Refusal is layered: a pre-LLM score gate handles the obvious empty/low-confidence
cases without spending an API call, and a prompt instruction lets the model itself
refuse when the provided context doesn't actually support an answer.
"""

from src.config import Config
from src.generation.llm_client import generate_structured
from src.generation.prompts import RESPONSE_SCHEMA, build_messages

NO_INFO_ANSWER = (
    "I don't have enough relevant information in the retrieved scikit-learn issues "
    "to answer this question."
)


def _pre_llm_refusal(reason: str, confidence_score: float | None) -> dict:
    return {
        "answer": NO_INFO_ANSWER,
        "citations": [],
        "refused": True,
        "refusal_reason": reason,
        "confidence_score": confidence_score,
    }


def generate(query: str, hits: list[dict], llm_client, config: Config) -> dict:
    generation_cfg = config.generation

    if not hits:
        return _pre_llm_refusal("no_hits", confidence_score=None)

    top_score = hits[0]["score"]
    if top_score < generation_cfg.min_top_score:
        return _pre_llm_refusal("low_confidence", confidence_score=top_score)

    messages = build_messages(query, hits)
    result = generate_structured(
        llm_client,
        messages,
        RESPONSE_SCHEMA,
        generation_cfg.model,
        generation_cfg.temperature,
    )

    if result["refused"]:
        return {
            "answer": result["answer"],
            "citations": [],
            "refused": True,
            "refusal_reason": "not_grounded",
            "confidence_score": top_score,
        }

    valid_issue_numbers = {hit["issue_number"] for hit in hits}
    citations = [c for c in result["citations"] if c in valid_issue_numbers]

    return {
        "answer": result["answer"],
        "citations": citations,
        "refused": False,
        "refusal_reason": None,
        "confidence_score": top_score,
    }
