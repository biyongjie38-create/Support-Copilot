from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models import SearchResult, TriageDecision


TRIAGE_ACTIONS = {"answer", "escalate", "general"}
LLM_UNAVAILABLE_REASON = "llm_unavailable"
LLM_CALL_FAILED_REASON = "llm_call_failed"


def build_chat_model() -> ChatOpenAI | None:
    settings = get_settings()
    if not settings.llm_enable_calls:
        return None
    api_key = settings.llm_api_key or ("ollama" if settings.llm_provider == "ollama" else None)
    if not api_key:
        return None
    return ChatOpenAI(
        model=settings.llm_chat_model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=0,
    )


def normalize_action(action: str) -> str:
    return action if action in TRIAGE_ACTIONS else "general"


def normalize_priority(priority: str) -> str:
    return priority if priority in {"low", "normal", "high", "urgent"} else "normal"


def unavailable_triage_decision(reason: str = LLM_UNAVAILABLE_REASON) -> TriageDecision:
    return TriageDecision(
        category="llm_unavailable",
        priority="high",
        action="general",
        reason=reason,
    )


def is_llm_unavailable_decision(decision: TriageDecision) -> bool:
    return decision.category == "llm_unavailable" or decision.reason in {
        LLM_UNAVAILABLE_REASON,
        LLM_CALL_FAILED_REASON,
    }


class RagAnswerComposer:
    def __init__(self, chain: Runnable | None = None) -> None:
        self.chain = chain or build_rag_answer_chain()

    def compose(self, question: str, documents: list[SearchResult]) -> str:
        payload = {"question": question, "context": format_documents_for_prompt(documents)}
        try:
            answer = str(self.chain.invoke(payload)).strip()
            return answer or fallback_grounded_answer(question, documents)
        except Exception:
            return fallback_grounded_answer(question, documents)


class TriageClassifier:
    def __init__(self, chain: Runnable | None = None, available: bool | None = None) -> None:
        if chain is not None:
            self.chain = chain
            self.available = True if available is None else available
            return
        llm = build_chat_model()
        self.available = llm is not None if available is None else available
        self.chain = build_triage_chain(llm) if llm is not None else None

    def classify(self, message: str) -> TriageDecision:
        if not self.available or self.chain is None:
            return unavailable_triage_decision()
        try:
            payload = self.chain.invoke({"message": message})
            if not isinstance(payload, dict):
                return unavailable_triage_decision(LLM_CALL_FAILED_REASON)
            raw_action = str(payload.get("action", "")).strip()
            if raw_action not in TRIAGE_ACTIONS:
                return unavailable_triage_decision(LLM_CALL_FAILED_REASON)
            return TriageDecision(
                category=str(payload.get("category", "general")),
                priority=normalize_priority(str(payload.get("priority", "normal"))),
                action=raw_action,
                reason=str(payload.get("reason", "由 LLM 完成初始分类")),
            )
        except Exception:
            return unavailable_triage_decision(LLM_CALL_FAILED_REASON)


class TriageRouter:
    def __init__(self, chain: Runnable | None = None, available: bool | None = None) -> None:
        if chain is not None:
            self.chain = chain
            self.available = True if available is None else available
            return
        llm = build_chat_model()
        self.available = llm is not None if available is None else available
        self.chain = build_triage_router_chain(llm) if llm is not None else None

    def choose_route(
        self,
        message: str,
        preliminary_decision: TriageDecision,
        documents: list[SearchResult],
    ) -> TriageDecision:
        if is_llm_unavailable_decision(preliminary_decision):
            return preliminary_decision
        if not self.available or self.chain is None:
            return unavailable_triage_decision()
        payload = {
            "message": message,
            "preliminary_decision": triage_payload_from_decision(preliminary_decision),
            "knowledge_candidates": format_documents_for_prompt(documents),
        }
        try:
            routed = self.chain.invoke(payload)
            if not isinstance(routed, dict):
                return unavailable_triage_decision(LLM_CALL_FAILED_REASON)
            raw_action = str(routed.get("action", "")).strip()
            if raw_action not in TRIAGE_ACTIONS:
                return unavailable_triage_decision(LLM_CALL_FAILED_REASON)
            return TriageDecision(
                category=str(routed.get("category", preliminary_decision.category)),
                priority=normalize_priority(str(routed.get("priority", preliminary_decision.priority))),
                action=raw_action,
                reason=str(routed.get("reason", "由 LLM 完成路由决策")),
            )
        except Exception:
            return unavailable_triage_decision(LLM_CALL_FAILED_REASON)


def build_rag_answer_chain(llm: ChatOpenAI | None = None) -> Runnable:
    llm = llm or build_chat_model()
    if llm is None:
        return RunnableLambda(lambda payload: fallback_grounded_answer_from_payload(payload))
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是客服知识库 RAG 助手。只能根据给定知识库片段回答，不要补充片段之外的事实。"
                "如果知识库没有覆盖问题，必须明确说明“知识库未覆盖该问题”。"
                "回答要简洁、可执行，并在句子中提到引用的文档标题。",
            ),
            ("human", "用户问题：{question}\n\n知识库片段：\n{context}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def build_triage_chain(llm: ChatOpenAI) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是客服 LangGraph Agent 的初始理解节点。只输出 JSON，不要输出 Markdown。"
                "字段必须包含 category、priority、action、reason。"
                "action 只能是 answer、escalate、general。"
                "你要先理解用户问题的真实意图，再给出一个初始判断。"
                "这个初始判断会被后续路由节点继续参考，但不要假设它一定是最终结果。",
            ),
            ("human", "{message}"),
        ]
    )
    return prompt | llm | JsonOutputParser()


def build_triage_router_chain(llm: ChatOpenAI) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是客服 LangGraph Agent 的路由决策节点。你必须根据用户原始问题、初始判断、"
                "以及知识库候选，选择最终路由。只输出 JSON，不要输出 Markdown。"
                "字段必须包含 category、priority、action、reason。"
                "action 只能是 answer、escalate、general。"
                "决策原则："
                "如果知识库候选已经足以直接回答用户问题，选择 answer；"
                "如果用户明确要求人工、表达投诉、涉及安全、账号被盗、支付异常、严重故障、"
                "或者需要人工核验与人工排查，选择 escalate；"
                "如果是寒暄、闲聊、范围外问题，或者当前候选仍不足以支撑回答，选择 general。"
                "不要把 action 机械地跟随初始判断，要重新理解用户意图后再决定最终路由。",
            ),
            (
                "human",
                "用户问题：{message}\n\n"
                "初始判断：{preliminary_decision}\n\n"
                "知识库候选：\n{knowledge_candidates}",
            ),
        ]
    )
    return prompt | llm | JsonOutputParser()


def format_documents_for_prompt(documents: list[SearchResult]) -> str:
    payload = [
        {
            "title": document.title,
            "source": document.source,
            "score": document.score,
            "content": document.content,
        }
        for document in documents
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def fallback_grounded_answer_from_payload(payload: dict[str, Any]) -> str:
    documents = []
    try:
        for item in json.loads(str(payload.get("context", "[]"))):
            documents.append(
                SearchResult(
                    chunk_id="",
                    document_id="",
                    title=str(item.get("title", "")),
                    source=str(item.get("source", "")),
                    content=str(item.get("content", "")),
                    score=float(item.get("score", 0.0)),
                )
            )
    except (TypeError, ValueError, json.JSONDecodeError):
        documents = []
    return fallback_grounded_answer(str(payload.get("question", "")), documents)


def fallback_grounded_answer(question: str, documents: list[SearchResult]) -> str:
    if not documents:
        return "知识库未覆盖该问题。请补充相关客服文档后再回答，或升级给人工支持处理。"
    primary = documents[0]
    snippets = []
    for document in documents[:3]:
        content = " ".join(document.content.split())
        snippets.append(f"《{document.title}》：{content[:260]}")
    return (
        f"根据知识库中与“{question}”相关的资料，优先参考《{primary.title}》。"
        f"\n\n可执行处理建议：\n- "
        + "\n- ".join(snippets)
    )


def triage_payload_from_decision(decision: TriageDecision) -> dict[str, str]:
    return {
        "category": decision.category,
        "priority": decision.priority,
        "action": decision.action,
        "reason": decision.reason,
    }
