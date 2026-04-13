from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app: str


class KnowledgeIngestResponse(BaseModel):
    inserted_documents: int
    document_titles: list[str]


class CitationRead(BaseModel):
    document_id: str
    title: str
    source: str
    snippet: str
    score: float


class RagQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class RagQueryResponse(BaseModel):
    answer: str
    citations: list[CitationRead]
    confidence: float = Field(ge=0, le=1)
    no_answer_reason: str | None = None


class TriageInvokeRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class TriageInvokeResponse(BaseModel):
    category: str
    priority: str
    action: Literal["answer", "escalate", "general"]
    answer: str
    citations: list[CitationRead]
    graph_trace: list[str]


class TriageGraphResponse(BaseModel):
    mermaid: str
    nodes: dict[str, str]
