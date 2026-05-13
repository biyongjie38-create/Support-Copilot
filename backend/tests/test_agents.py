import pytest
import json
from pathlib import Path

from app.agent.rag import AgenticRagAgent
from app.agent.rag_graph import SupportRagGraph
from app.agent.triage import LangGraphTriageAgent
from app.config import get_settings
from app.data.demo_knowledge import DEMO_KNOWLEDGE
from app.models import TriageDecision
from app.postgres_store import InMemoryKnowledgeStore


@pytest.fixture(autouse=True)
def disable_live_llm_calls(monkeypatch):
    monkeypatch.setenv("LLM_ENABLE_CALLS", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def build_store() -> InMemoryKnowledgeStore:
    store = InMemoryKnowledgeStore()
    store.ingest_documents(DEMO_KNOWLEDGE)
    return store


def load_eval_cases():
    return json.loads((Path(__file__).parents[1] / "eval" / "eval_cases.json").read_text(encoding="utf-8"))


class StubClassifier:
    def __init__(self, decision: TriageDecision):
        self.decision = decision

    def classify(self, message: str) -> TriageDecision:
        return self.decision


class StubRouter:
    def __init__(self, decision: TriageDecision):
        self.decision = decision

    def choose_route(self, message: str, preliminary_decision: TriageDecision, documents):
        return self.decision


class DocsAwareRouter:
    def choose_route(self, message: str, preliminary_decision: TriageDecision, documents):
        if documents:
            return TriageDecision(
                category=preliminary_decision.category,
                priority=preliminary_decision.priority,
                action="answer",
                reason="router saw relevant knowledge candidates",
            )
        return TriageDecision(
            category=preliminary_decision.category,
            priority=preliminary_decision.priority,
            action="general",
            reason="router saw no relevant knowledge candidates after rag grading",
        )


def test_rag_hit_returns_citation():
    result = AgenticRagAgent(build_store(), min_score=0.05).query("密码重置")

    assert result.no_answer_reason is None
    assert result.citations
    assert result.citations[0].source == "support-handbook/account-password-reset.md"


def test_rag_api_429_question_returns_api_rate_limit_citation():
    result = AgenticRagAgent(build_store()).query("API 一直返回 429 是什么意思？")

    assert result.no_answer_reason is None
    assert result.citations
    assert any("api-rate-limit" in citation.source for citation in result.citations)


def test_rag_no_hit_does_not_hallucinate():
    result = AgenticRagAgent(build_store(), min_score=0.4).query("bluetooth headset tv")

    assert result.no_answer_reason == "no_relevant_knowledge"
    assert result.citations == []
    assert "知识库未覆盖" in result.answer


def test_triage_uses_local_router_when_llm_is_disabled():
    result = LangGraphTriageAgent(build_store()).invoke("API 429 meaning")

    assert result.action == "answer"
    assert result.citations
    assert any("api-rate-limit" in citation.source for citation in result.citations)


def test_triage_uses_router_decision_for_knowledge_question():
    classifier = StubClassifier(
        TriageDecision(
            category="api_issue",
            priority="high",
            action="escalate",
            reason="initial classifier thought this needed escalation",
        )
    )
    router = StubRouter(
        TriageDecision(
            category="api_issue",
            priority="normal",
            action="answer",
            reason="router decided the knowledge base can answer the question",
        )
    )

    result = LangGraphTriageAgent(build_store(), classifier=classifier, router=router).invoke("API 429 meaning")

    assert result.action == "answer"
    assert result.citations
    assert any("api-rate-limit" in citation.source for citation in result.citations)


def test_triage_keeps_router_escalation_for_explicit_human_request():
    classifier = StubClassifier(
        TriageDecision(
            category="billing_complaint",
            priority="high",
            action="answer",
            reason="initial classifier was optimistic",
        )
    )
    router = StubRouter(
        TriageDecision(
            category="billing_complaint",
            priority="high",
            action="escalate",
            reason="router detected an explicit human escalation request",
        )
    )

    result = LangGraphTriageAgent(
        build_store(),
        classifier=classifier,
        router=router,
    ).invoke("duplicate charge, transfer me to a human")

    assert result.action == "escalate"
    assert result.citations == []


def test_triage_routes_with_the_same_rag_threshold_used_for_final_answer():
    classifier = StubClassifier(
        TriageDecision(
            category="api_issue",
            priority="normal",
            action="answer",
            reason="initial classifier thinks this is answerable",
        )
    )
    agent = LangGraphTriageAgent(
        build_store(),
        classifier=classifier,
        router=DocsAwareRouter(),
        rag_agent=AgenticRagAgent(build_store(), min_score=0.95),
    )

    result = agent.invoke("API 429 meaning")

    assert result.action == "general"
    assert result.citations == []


@pytest.mark.parametrize("case", [case for case in load_eval_cases() if case["expected_action"] == "answer"])
def test_rag_eval_answer_cases_retrieve_expected_source(case):
    result = AgenticRagAgent(build_store()).query(case["question"])

    assert result.no_answer_reason is None
    assert result.citations
    assert any(case["expected_source"] == citation.source for citation in result.citations)


@pytest.mark.parametrize("case", load_eval_cases())
def test_triage_eval_cases_follow_expected_action_and_source(case):
    result = LangGraphTriageAgent(build_store()).invoke(case["question"])

    assert result.action == case["expected_action"]
    if case["expected_source"]:
        assert any(case["expected_source"] == citation.source for citation in result.citations)
    else:
        assert result.citations == []


def test_support_rag_graph_preserves_retrieval_state_and_metadata():
    graph = SupportRagGraph(build_store())

    state = graph.invoke("API 一直返回 429 是什么意思？")
    mermaid, nodes = graph.graph_metadata()

    assert state["retrieved_docs"]
    assert state["graded_docs"]
    assert state["citations"]
    assert state["no_answer_reason"] is None
    assert state["rewrite_count"] == 0
    assert state["route_reason"]
    assert "generate_query_or_respond" in mermaid
    assert "retrieve" in mermaid
    assert "grade_documents" in mermaid
    assert "generate_query_or_respond" in nodes
