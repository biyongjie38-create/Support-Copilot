from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from app.config import get_settings
from app.llm_router import RagAnswerComposer, build_chat_model
from app.models import Citation, RagResult, SearchResult
from app.postgres_store import KnowledgeStore


SUPPORT_RAG_GRAPH_NODES = {
    "generate_query_or_respond": "决定是否调用知识库检索工具，并生成本轮检索 query。",
    "retrieve": "通过 ToolNode 调用客服知识库检索工具。",
    "grade_documents": "根据相关性阈值筛选 retrieved_docs，决定回答、改写或拒答。",
    "rewrite_query": "当首轮证据不足时改写检索 query，并限制改写次数。",
    "generate_answer": "严格基于 graded_docs 生成答案、引用和置信度。",
    "final": "统一输出答案、citations、no_answer_reason 与 graph state。",
}


class SupportRagState(TypedDict, total=False):
    question: str
    top_k: int
    messages: Annotated[list[BaseMessage], add_messages]
    search_query: str
    retrieved_docs: list[SearchResult]
    graded_docs: list[SearchResult]
    rewrite_count: int
    route_reason: str
    answer: str
    citations: list[Citation]
    confidence: float
    no_answer_reason: str | None
    result: RagResult


RouteAfterGrade = Literal["generate_answer", "rewrite_query", "final"]


class SupportRagGraph:
    """LangGraph Agentic RAG workflow for support knowledge-grounded answers."""

    def __init__(
        self,
        store: KnowledgeStore,
        answer_composer: RagAnswerComposer | None = None,
        min_score: float | None = None,
        max_rewrites: int = 1,
    ) -> None:
        settings = get_settings()
        self.store = store
        self.answer_composer = answer_composer or RagAnswerComposer()
        self.min_score = settings.rag_min_score if min_score is None else min_score
        self.max_rewrites = max_rewrites
        self.retrieve_tool = self._build_retrieve_tool()
        self.tool_model = self._build_tool_model()
        self.graph = self._build_graph()

    def invoke(self, question: str, top_k: int = 5) -> SupportRagState:
        return self.graph.invoke(
            {
                "question": question,
                "top_k": top_k,
                "messages": [HumanMessage(content=question)],
                "search_query": question,
                "retrieved_docs": [],
                "graded_docs": [],
                "rewrite_count": 0,
                "route_reason": "start",
                "citations": [],
                "confidence": 0.0,
                "no_answer_reason": None,
            }
        )

    def query(self, question: str, top_k: int = 5) -> RagResult:
        state = self.invoke(question, top_k=top_k)
        result = state.get("result")
        if result is not None:
            return result
        return RagResult(
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            confidence=state.get("confidence", 0.0),
            no_answer_reason=state.get("no_answer_reason"),
        )

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        return self._grade_documents(self.store.search_knowledge(question, limit=top_k))

    def graph_metadata(self) -> tuple[str, dict[str, str]]:
        return self.graph.get_graph().draw_mermaid(), SUPPORT_RAG_GRAPH_NODES

    def _build_graph(self):
        workflow = StateGraph(SupportRagState)
        workflow.add_node("generate_query_or_respond", self._generate_query_or_respond_node)
        workflow.add_node("retrieve", ToolNode([self.retrieve_tool]))
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("final", self._final_node)

        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {"tools": "retrieve", END: "final"},
        )
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self._route_after_grade,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "final": "final",
            },
        )
        workflow.add_edge("rewrite_query", "generate_query_or_respond")
        workflow.add_edge("generate_answer", "final")
        workflow.add_edge("final", END)
        return workflow.compile()

    def _build_retrieve_tool(self):
        store = self.store

        @tool
        def retrieve_support_knowledge(query: str, limit: int = 5) -> str:
            """Search the support knowledge base for relevant policy and troubleshooting excerpts."""

            documents = store.search_knowledge(query, limit=limit)
            return json.dumps([self._serialize_search_result(document) for document in documents], ensure_ascii=False)

        return retrieve_support_knowledge

    def _build_tool_model(self):
        llm = build_chat_model()
        if llm is None:
            return None
        try:
            return llm.bind_tools([self.retrieve_tool])
        except Exception:
            return None

    def _generate_query_or_respond_node(self, state: SupportRagState) -> SupportRagState:
        query = state.get("search_query") or state["question"]
        top_k = state.get("top_k", 5)
        if self.tool_model is not None:
            try:
                response = self.tool_model.invoke(
                    [
                        SystemMessage(
                            content=(
                                "你是客服知识库 RAG agent。除非用户明显闲聊或范围外，否则必须调用 "
                                "retrieve_support_knowledge 工具检索证据；不要凭记忆回答客服政策。"
                            )
                        ),
                        *state.get("messages", []),
                    ]
                )
                if getattr(response, "tool_calls", None):
                    return {
                        "messages": [response],
                        "route_reason": "generate_query_or_respond: LLM selected retrieval tool",
                    }
                return {
                    "messages": [response],
                    "route_reason": "generate_query_or_respond: LLM responded without retrieval",
                    "no_answer_reason": "no_retrieval_requested",
                }
            except Exception:
                pass

        tool_call = {
            "name": self.retrieve_tool.name,
            "args": {"query": query, "limit": top_k},
            "id": f"retrieve-{uuid.uuid4()}",
        }
        return {
            "messages": [AIMessage(content="", tool_calls=[tool_call])],
            "search_query": query,
            "route_reason": f"generate_query_or_respond: fallback tool query={query}",
        }

    def _grade_documents_node(self, state: SupportRagState) -> SupportRagState:
        retrieved = self._parse_latest_tool_documents(state)
        graded = self._grade_documents(retrieved)
        if graded:
            return {
                "retrieved_docs": retrieved,
                "graded_docs": graded,
                "route_reason": f"grade_documents: accepted {len(graded)} of {len(retrieved)} retrieved documents",
                "no_answer_reason": None,
            }
        if state.get("rewrite_count", 0) < self.max_rewrites:
            return {
                "retrieved_docs": retrieved,
                "graded_docs": [],
                "route_reason": "grade_documents: evidence too weak, rewriting query",
                "no_answer_reason": "needs_query_rewrite",
            }
        return {
            "retrieved_docs": retrieved,
            "graded_docs": [],
            "route_reason": "grade_documents: evidence too weak after rewrite budget",
            "no_answer_reason": "no_relevant_knowledge",
        }

    def _route_after_grade(self, state: SupportRagState) -> RouteAfterGrade:
        if state.get("graded_docs"):
            return "generate_answer"
        if state.get("no_answer_reason") == "needs_query_rewrite":
            return "rewrite_query"
        return "final"

    def _rewrite_query_node(self, state: SupportRagState) -> SupportRagState:
        rewritten = self._rewrite_query(state["question"], state.get("retrieved_docs", []))
        return {
            "search_query": rewritten,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "route_reason": f"rewrite_query: {rewritten}",
        }

    def _generate_answer_node(self, state: SupportRagState) -> SupportRagState:
        documents = state.get("graded_docs", [])
        answer = self.answer_composer.compose(state["question"], documents)
        citations = [self._to_citation(document) for document in documents]
        confidence = self._confidence(documents)
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "no_answer_reason": None,
            "messages": [AIMessage(content=answer)],
            "route_reason": f"generate_answer: generated answer with {len(citations)} citations",
        }

    def _final_node(self, state: SupportRagState) -> SupportRagState:
        no_answer_reason = state.get("no_answer_reason")
        answer = state.get("answer")
        citations = state.get("citations", [])
        confidence = state.get("confidence", 0.0)
        if not answer or no_answer_reason:
            no_answer_reason = "no_relevant_knowledge" if no_answer_reason != "no_retrieval_requested" else no_answer_reason
            answer = (
                "知识库未覆盖该问题。为了避免编造答案，我不会基于猜测回复；"
                "建议补充相关文档或升级给人工支持处理。"
            )
            citations = []
            confidence = 0.0
        result = RagResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            no_answer_reason=no_answer_reason,
        )
        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "no_answer_reason": no_answer_reason,
            "result": result,
            "route_reason": f"final: {no_answer_reason or 'answered'}",
        }

    def _parse_latest_tool_documents(self, state: SupportRagState) -> list[SearchResult]:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, ToolMessage):
                try:
                    payload = json.loads(str(message.content))
                except json.JSONDecodeError:
                    return []
                return [self._deserialize_search_result(item) for item in payload if isinstance(item, dict)]
        return []

    def _grade_documents(self, documents: list[SearchResult]) -> list[SearchResult]:
        return [document for document in documents if document.score >= self.min_score]

    def _rewrite_query(self, question: str, documents: list[SearchResult]) -> str:
        normalized = question.lower()
        expansions = []
        if "429" in normalized or "限流" in question or "请求频率" in question:
            expansions.append("API 429 限流 请求频率 超过 套餐限制 指数退避 request_id")
        if "退钱" in question or "退款" in question or "扣费" in question:
            expansions.append("订阅 退款 重复扣费 支付流水号 原路退款")
        if "发票" in question:
            expansions.append("发票 专用发票 抬头 税号 开票审核")
        if "导出" in question or "下载" in question:
            expansions.append("数据导出 下载 会话记录 管理员")
        if "锁" in question or "登录" in question or "密码" in question:
            expansions.append("账号锁定 密码重置 安全核验")
        if "删除" in question or "隐私" in question:
            expansions.append("删除账号 隐私数据 冷静期 合规保留")
        if documents:
            expansions.extend(document.title for document in documents[:2] if document.score >= self.min_score * 0.5)
        return " ".join([question, *expansions]).strip()

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

    def _serialize_search_result(self, document: SearchResult) -> dict[str, Any]:
        return {
            "chunk_id": document.chunk_id,
            "document_id": document.document_id,
            "title": document.title,
            "source": document.source,
            "content": document.content,
            "score": document.score,
        }

    def _deserialize_search_result(self, payload: dict[str, Any]) -> SearchResult:
        return SearchResult(
            chunk_id=str(payload.get("chunk_id", "")),
            document_id=str(payload.get("document_id", "")),
            title=str(payload.get("title", "")),
            source=str(payload.get("source", "")),
            content=str(payload.get("content", "")),
            score=float(payload.get("score", 0.0)),
        )
