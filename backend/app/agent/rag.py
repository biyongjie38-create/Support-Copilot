from __future__ import annotations

from app.config import get_settings
from app.llm_router import RagAnswerComposer
from app.models import Citation, RagResult, SearchResult
from app.postgres_store import KnowledgeStore


class AgenticRagAgent:
    """Retrieve, grade, and answer strictly from the support knowledge base."""

    def __init__(
        self,
        store: KnowledgeStore,
        answer_composer: RagAnswerComposer | None = None,
        min_score: float | None = None,
    ) -> None:
        settings = get_settings()
        self.store = store
        self.answer_composer = answer_composer or RagAnswerComposer()
        self.min_score = settings.rag_min_score if min_score is None else min_score

    def query(self, question: str, top_k: int = 5) -> RagResult:
        relevant = self.retrieve(question, top_k=top_k)
        if not relevant:
            return RagResult(
                answer="知识库未覆盖该问题。为了避免编造答案，我不会基于猜测回复；建议补充相关文档或升级给人工支持处理。",
                citations=[],
                confidence=0.0,
                no_answer_reason="no_relevant_knowledge",
            )
        answer = self.answer_composer.compose(question, relevant)
        citations = [self._to_citation(document) for document in relevant]
        return RagResult(
            answer=answer,
            citations=citations,
            confidence=self._confidence(relevant),
            no_answer_reason=None,
        )

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        retrieved = self.store.search_knowledge(question, limit=top_k)
        return self._grade_documents(retrieved)

    def _grade_documents(self, documents: list[SearchResult]) -> list[SearchResult]:
        return [document for document in documents if document.score >= self.min_score]

    def _confidence(self, documents: list[SearchResult]) -> float:
        if not documents:
            return 0.0
        top_score = max(document.score for document in documents)
        mean_score = sum(document.score for document in documents) / len(documents)
        return round(min(1.0, top_score * 0.7 + mean_score * 0.3), 4)

    def _to_citation(self, document: SearchResult) -> Citation:
        snippet = " ".join(document.content.split())[:320]
        return Citation(
            document_id=document.document_id,
            title=document.title,
            source=document.source,
            snippet=snippet,
            score=round(document.score, 4),
        )
