"""FastAPI app exposing the RAG pipeline over HTTP.

GET /health -- liveness check for whatever hosts this.
POST /query -- retrieves chunks (baseline or hybrid mode, per the request),
generates a grounded answer, and returns a lean public response: answer,
citations, and refusal info only. No raw scores or chunk text -- that's
implementation detail for the eval harness, not the public API.
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.deps import build_app_state, get_app_state
from src.api.rate_limit import limiter
from src.api.schemas import Citation, QueryRequest, QueryResponse, RetrievalMode
from src.config import load_config
from src.generation.generate import generate
from src.retrieval.filter_extraction import extract_filters
from src.retrieval.retrieve import ISSUE_URL_TEMPLATE, retrieve

# uvicorn imports this module directly (no main()/if __name__ guard to call
# load_dotenv() from, unlike the other entrypoints), so it's loaded here at
# import time instead -- harmless in prod, where real env vars are already set
# by the host and no .env file exists.
load_dotenv()

# Loaded at import time (config.yaml has no secrets, so this doesn't need
# dotenv/env vars) so the rate-limit string is available to decorate /query
# below, before the app's lifespan/startup has run.
_config = load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.app_state = build_app_state()
    yield


app = FastAPI(lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=502,
        content={"detail": "Upstream service error. Please try again."},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


def _filters_for_mode(mode: RetrievalMode, query: str) -> dict:
    return {} if mode == RetrievalMode.baseline else extract_filters(query)


@app.post("/query", response_model=QueryResponse)
@limiter.limit(_config.api.rate_limit)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    app_state = get_app_state(request)

    filters = _filters_for_mode(body.retrieval_mode, body.question)
    hits = retrieve(
        body.question,
        filters,
        app_state.qdrant_client,
        app_state.openai_client,
        app_state.config,
    )
    result = generate(body.question, hits, app_state.openai_client, app_state.config)

    citations = [
        Citation(
            issue_number=issue_number,
            url=ISSUE_URL_TEMPLATE.format(issue_number=issue_number),
        )
        for issue_number in result["citations"]
    ]

    return QueryResponse(
        answer=result["answer"],
        citations=citations,
        refused=result["refused"],
        refusal_reason=result["refusal_reason"],
    )
