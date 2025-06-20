"""Pydantic models."""

from pydantic import BaseModel
from typing import List


class MessageData(BaseModel):
    """Datamodel for messages."""

    query: str
    chat_history: List[dict] | None
    model_name: str
    temperature: float
    top_p: float
    max_tokens: int


from typing import List
from pydantic import BaseModel

class RAGResponse(BaseModel):
    answer: str
    context: List[str]

    question: str = ""
    retrieved_contexts: List[str] = []

class QAItem(BaseModel):
    question: str
    answer: str


class QAList(BaseModel):
    """Datamodel for trainset."""

    items: List[QAItem]
    model_name: str
    temperature: float
    top_p: float
    max_tokens: int

class RetrieveRequest(BaseModel):
    query: str
    k: int = 5

class RetrieveResponse(BaseModel):
    retrieved_docs: List[str]
    k :int