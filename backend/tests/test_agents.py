from app.agent.rag import AgenticRagAgent
from app.agent.triage import LangGraphTriageAgent
from app.data.demo_knowledge import DEMO_KNOWLEDGE
from app.postgres_store import InMemoryKnowledgeStore


def build_store() -> InMemoryKnowledgeStore:
    store = InMemoryKnowledgeStore()
    store.ingest_documents(DEMO_KNOWLEDGE)
    return store


def test_rag_hit_returns_citation():
    result = AgenticRagAgent(build_store(), min_score=0.05).query("怎么重置密码？")

    assert result.no_answer_reason is None
    assert result.citations
    assert result.citations[0].source == "support-handbook/account-password-reset.md"


def test_rag_no_hit_does_not_hallucinate():
    result = AgenticRagAgent(build_store(), min_score=0.4).query("蓝牙耳机怎么连接电视？")

    assert result.no_answer_reason == "no_relevant_knowledge"
    assert result.citations == []
    assert "知识库未覆盖" in result.answer


def test_triage_routes_knowledge_question_to_rag():
    result = LangGraphTriageAgent(build_store()).invoke("API 一直返回 429 是什么意思？")

    assert result.action == "answer"
    assert result.citations
    assert any("api-rate-limit" in citation.source for citation in result.citations)
    assert any(step.startswith("rag_answer") for step in result.graph_trace)


def test_triage_routes_complaint_to_escalation():
    result = LangGraphTriageAgent(build_store()).invoke("我要投诉你们重复扣费，请转人工")

    assert result.action == "escalate"
    assert result.priority in {"high", "urgent"}
    assert result.citations == []
    assert any(step.startswith("escalate") for step in result.graph_trace)
