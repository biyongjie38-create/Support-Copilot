from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class KnowledgeDocument:
    id: str
    slug: str
    title: str
    source: str
    content: str
    created_at: datetime = field(default_factory=utc_now)


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    document_id: str
    title: str
    source: str
    content: str
    score: float


@dataclass(slots=True)
class Citation:
    document_id: str
    title: str
    source: str
    snippet: str
    score: float


@dataclass(slots=True)
class RagResult:
    answer: str
    citations: list[Citation]
    confidence: float
    no_answer_reason: str | None = None


TriageAction = Literal["answer", "escalate", "general"]


@dataclass(slots=True)
class TriageDecision:
    category: str
    priority: str
    action: TriageAction
    reason: str


@dataclass(slots=True)
class TriageResult:
    category: str
    priority: str
    action: TriageAction
    answer: str
    citations: list[Citation]
    graph_trace: list[str]
