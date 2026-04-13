from __future__ import annotations

import json
from typing import Any

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.models import SearchResult, TriageDecision


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
    def __init__(self, chain: Runnable | None = None) -> None:
        self.chain = chain or build_triage_chain()

    def classify(self, message: str) -> TriageDecision:
        try:
            payload = self.chain.invoke({"message": message})
            if not isinstance(payload, dict):
                return fallback_triage(message)
            return TriageDecision(
                category=str(payload.get("category", "general")),
                priority=normalize_priority(str(payload.get("priority", "normal"))),
                action=normalize_action(str(payload.get("action", "general"))),
                reason=str(payload.get("reason", "")),
            )
        except Exception:
            return fallback_triage(message)


def build_rag_answer_chain() -> Runnable:
    llm = build_chat_model()
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


def build_triage_chain() -> Runnable:
    llm = build_chat_model()
    if llm is None:
        return RunnableLambda(lambda payload: triage_payload_from_decision(fallback_triage(str(payload["message"]))))
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是客服 LangGraph 分诊 Agent 的分类节点。只输出 JSON，不要输出 Markdown。"
                "字段必须包含 category、priority、action、reason。"
                "action 只能是 answer、escalate、general："
                "知识库可回答的问题用 answer；投诉、人工、安全、严重故障用 escalate；普通寒暄或范围外问题用 general。"
                "priority 只能是 low、normal、high、urgent。",
            ),
            ("human", "{message}"),
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


def fallback_triage(message: str) -> TriageDecision:
    lowered = message.lower()
    urgent_terms = ["p0", "宕机", "崩了", "数据丢失", "支付不可用", "账号被盗"]
    escalate_terms = ["投诉", "人工", "转人工", "客服经理", "被盗", "安全", "严重", "重复扣费"]
    knowledge_terms = [
        "怎么",
        "如何",
        "密码",
        "登录",
        "退款",
        "发票",
        "导出",
        "订阅",
        "取消",
        "删除账号",
        "隐私",
        "限流",
        "429",
        "api",
        "锁定",
        "报错",
    ]
    general_strong_terms = ["天气", "笑话"]
    greeting_terms = ["你好", "hello", "hi"]
    has_knowledge_intent = any(term in lowered or term in message for term in knowledge_terms)
    if any(term in lowered or term in message for term in urgent_terms):
        return TriageDecision("incident_or_security", "urgent", "escalate", "命中严重故障或安全风险关键词")
    if any(term in lowered or term in message for term in escalate_terms):
        return TriageDecision("human_escalation", "high", "escalate", "用户表达投诉、人工或安全诉求")
    if any(term in lowered or term in message for term in general_strong_terms):
        return TriageDecision("general", "low", "general", "普通寒暄或客服知识库范围外问题")
    if any(term in lowered or term in message for term in greeting_terms) and not has_knowledge_intent:
        return TriageDecision("general", "low", "general", "普通寒暄或客服知识库范围外问题")
    if has_knowledge_intent:
        return TriageDecision("knowledge_question", "normal", "answer", "问题适合先检索客服知识库")
    return TriageDecision("general", "low", "general", "未命中知识库或升级意图")


def triage_payload_from_decision(decision: TriageDecision) -> dict[str, str]:
    return {
        "category": decision.category,
        "priority": decision.priority,
        "action": decision.action,
        "reason": decision.reason,
    }


def normalize_action(action: str) -> str:
    return action if action in {"answer", "escalate", "general"} else "general"


def normalize_priority(priority: str) -> str:
    return priority if priority in {"low", "normal", "high", "urgent"} else "normal"
