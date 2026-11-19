from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.cache import QueryCache
from src.generator import AnswerGenerator
from src.models import MinimalAnswer, MinimalSearchResults
from src.retriever import BM25Retriever

app = FastAPI(
    title="RAG Against the Machine API",
    description="Local HTTP API for codebase retrieval and answer generation.",
    version="1.0.0",
)

_retriever: Optional[BM25Retriever] = None
_generator: Optional[AnswerGenerator] = None
_cache = QueryCache()


class AnswerRequest(BaseModel):
    """Request payload for the /answer endpoint."""
    query: str
    k: int = 5


def get_retriever() -> BM25Retriever:
    """Lazy-loads and caches the BM25Retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever()
    return _retriever


def get_generator() -> AnswerGenerator:
    """Lazy-loads and caches the AnswerGenerator (LLM) instance."""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


@app.get("/health")
def health() -> dict[str, str]:
    """Simple healthcheck endpoint."""
    return {"status": "ok"}


@app.get("/search", response_model=MinimalSearchResults)
def search_endpoint(
    query: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
) -> MinimalSearchResults:
    """Retrieves top-k source chunks for a query."""
    try:
        retriever = get_retriever()
        return retriever.search_single(query_id="api-query", query=query, k=k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/answer", response_model=MinimalAnswer)
def answer_endpoint(payload: AnswerRequest) -> MinimalAnswer:
    """Retrieves context and generates an answer using the local LLM."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if payload.k <= 0:
        raise HTTPException(status_code=400, detail="k must be greater than 0.")

    try:
        retriever = get_retriever()
        search_res = retriever.search_single(
            query_id="api-query", query=payload.query, k=payload.k
        )

        cached_answer = _cache.get("answer", payload.query, payload.k)
        if isinstance(cached_answer, str):
            return MinimalAnswer(
                question_id=search_res.question_id,
                question=search_res.question,
                retrieved_sources=search_res.retrieved_sources,
                answer=cached_answer,
            )

        generator = get_generator()
        answer_text = generator._generate_single_answer(
            search_res.question, search_res.retrieved_sources
        )
        _cache.set("answer", payload.query, payload.k, answer_text)

        return MinimalAnswer(
            question_id=search_res.question_id,
            question=search_res.question,
            retrieved_sources=search_res.retrieved_sources,
            answer=answer_text,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
