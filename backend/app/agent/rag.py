from __future__ import annotations

from app.llm_router import RagAnswerComposer
from app.models import RagResult, SearchResult
from app.postgres_store import KnowledgeStore
from app.agent.rag_graph import SupportRagGraph


class AgenticRagAgent:
    """Compatibility wrapper around the LangGraph SupportRagGraph workflow."""

    def __init__(
        self,
        store: KnowledgeStore,
        answer_composer: RagAnswerComposer | None = None,
        min_score: float | None = None,
    ) -> None:
        self.graph = SupportRagGraph(store, answer_composer=answer_composer, min_score=min_score)

    def query(self, question: str, top_k: int = 5) -> RagResult:
        return self.graph.query(question, top_k=top_k)

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        return self.graph.retrieve(question, top_k=top_k)
