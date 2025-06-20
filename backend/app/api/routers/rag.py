"""Endpoints."""

from fastapi import APIRouter
from app.utils.models import MessageData, RAGResponse, QAList,RetrieveRequest, RetrieveResponse
from app.utils.rag_functions import (
    get_zero_shot_query,
    get_compiled_rag,
    compile_rag,
    get_list_openai_models,retrieve_only
)

rag_router = APIRouter(prefix="/rag", tags=["rag"])

@rag_router.get("/healthcheck")
async def healthcheck():
    return {"message": "Service is up."}

@rag_router.post("/retrieve-only", response_model=RetrieveResponse)
async def retrieve_only_endpoint(payload: RetrieveRequest):
    docs = retrieve_only(query=payload.query, k=payload.k)
    return RetrieveResponse(retrieved_docs=docs)

@rag_router.get("/list-models")
async def list_models():
    return {"models": get_list_openai_models()}

@rag_router.post("/zero-shot-query", response_model=RAGResponse)
async def zero_shot_query(payload: MessageData):
    return get_zero_shot_query(
        query=payload.query,
        model_name=payload.model_name,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
    )

@rag_router.post("/compiled-query", response_model=RAGResponse)
async def compiled_query(payload: MessageData):
    return get_compiled_rag(
        query=payload.query,
        model_name=payload.model_name,
        temperature=payload.temperature,
        top_p=payload.top_p,
        max_tokens=payload.max_tokens,
        k=payload.k,
    )

@rag_router.post("/compile-program")
async def compile_program(qa_list: QAList):
    return compile_rag(
        items=qa_list.items,
        model_name=qa_list.model_name,
        temperature=qa_list.temperature,
        top_p=qa_list.top_p,
        max_tokens=qa_list.max_tokens,
        k=qa_list.k,
    )