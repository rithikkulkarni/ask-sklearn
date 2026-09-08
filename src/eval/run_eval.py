"""Evaluation harness: runs the hand-written test set (data/eval/test_set.json)
through both baseline (pure vector) and hybrid (vector + extracted filters)
retrieval, scores each question with retrieval precision/recall, groundedness,
and rubric-based correctness, and persists the full per-question transcript
plus an aggregate summary for each mode to data/eval/runs/.

Usage:
    python -m src.eval.run_eval --smoke   # fast hand-picked iteration subset
    python -m src.eval.run_eval --full    # full 30-50 question validation set
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.config import load_config
from src.embedding.qdrant_store import get_client
from src.eval import judge_client
from src.eval.aggregate import build_summary
from src.eval.dataset import EvalQuestion, load_questions
from src.eval.judge import citation_validity_rate, judge_correctness, judge_groundedness
from src.eval.retrieval_metrics import precision_recall
from src.generation.generate import generate
from src.retrieval.filter_extraction import extract_filters
from src.retrieval.retrieve import retrieve

RUNS_DIR = Path("data/eval/runs")
MODES = ["baseline", "hybrid"]


def _filters_for_mode(mode: str, query: str) -> dict:
    return {} if mode == "baseline" else extract_filters(query)


def run_question(
    question: EvalQuestion, mode: str, qdrant_client, openai_client, config
) -> dict:
    filters = _filters_for_mode(mode, question.query)
    hits = retrieve(question.query, filters, qdrant_client, openai_client, config)
    result = generate(question.query, hits, openai_client, config)

    retrieved_issue_numbers = sorted({hit["issue_number"] for hit in hits})
    retrieval_metrics = precision_recall(
        retrieved_issue_numbers, question.target_issue_numbers
    )

    groundedness = None
    if not result["refused"]:
        groundedness = judge_groundedness(
            result["answer"],
            hits,
            config.eval.judge_model,
            config.eval.judge_temperature,
        )

    correctness = judge_correctness(
        question.query,
        result["answer"],
        question.rubric_facts,
        config.eval.judge_model,
        config.eval.judge_temperature,
    )

    return {
        "question_id": question.id,
        "query": question.query,
        "mode": mode,
        "tags": question.tags,
        "target_issue_numbers": question.target_issue_numbers,
        "retrieved_issue_numbers": retrieved_issue_numbers,
        "retrieval_metrics": retrieval_metrics,
        "answer": result["answer"],
        "citations": result["citations"],
        "refused": result["refused"],
        "refusal_reason": result["refusal_reason"],
        "confidence_score": result["confidence_score"],
        "citation_validity_rate": citation_validity_rate(result["citations"], hits),
        "groundedness": groundedness,
        "correctness": correctness,
    }


def run(questions: list[EvalQuestion], qdrant_client, openai_client, config) -> dict:
    records_by_mode = {mode: [] for mode in MODES}
    for i, question in enumerate(questions, start=1):
        for mode in MODES:
            record = run_question(question, mode, qdrant_client, openai_client, config)
            records_by_mode[mode].append(record)
        print(f"[{i}/{len(questions)}] {question.id} done")
    return records_by_mode


def main():
    parser = argparse.ArgumentParser(
        description="Run the retrieval/generation eval harness."
    )
    subset = parser.add_mutually_exclusive_group(required=True)
    subset.add_argument(
        "--smoke", action="store_true", help="Run only the fast smoke-test subset."
    )
    subset.add_argument(
        "--full", action="store_true", help="Run the full question set."
    )
    args = parser.parse_args()

    load_dotenv()
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set (add it to .env)")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY not set (add it to .env)")
    judge_client.configure(gemini_key)

    config = load_config()
    openai_client = OpenAI(api_key=openai_key)
    qdrant_client = get_client(
        config.vector_store.host,
        config.vector_store.port,
        url=os.environ.get("QDRANT_URL"),
        api_key=os.environ.get("QDRANT_API_KEY"),
    )

    questions = load_questions(smoke_only=args.smoke)
    subset_label = "smoke" if args.smoke else "full"
    print(f"Running eval on {len(questions)} questions ({subset_label} set)...")

    records_by_mode = run(questions, qdrant_client, openai_client, config)
    summary = {
        mode: build_summary(records) for mode, records in records_by_mode.items()
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{subset_label}"
    out_path = RUNS_DIR / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"run_id": run_id, "summary": summary, "records": records_by_mode},
            f,
            indent=2,
        )

    print(f"\nWrote results to {out_path}")
    for mode, mode_summary in summary.items():
        print(f"\n[{mode}]")
        print(json.dumps(mode_summary, indent=2))


if __name__ == "__main__":
    main()
