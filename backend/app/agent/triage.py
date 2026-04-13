from __future__ import annotations

from dataclasses import replace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.rag import AgenticRagAgent
from app.llm_router import TriageClassifier
from app.models import TriageDecision, TriageResult
from app.postgres_store import KnowledgeStore


TRIAGE_GRAPH_MERMAID = """flowchart TD
    intake["intake: 接收用户问题"] --> classify["classify: 判断问题类别、优先级和动作"]
    classify --> route["route: 按动作选择处理路径"]
    route -->|answer| rag_answer["rag_answer: 调用 Agentic RAG"]
    route -->|escalate| escalate["escalate: 生成人工升级摘要"]
    route -->|general| general_response["general_response: 给出范围说明"]
    rag_answer --> final["final: 输出统一结果"]
    escalate --> final
    general_response --> final
"""


TRIAGE_GRAPH_NODES = {
    "intake": "接收用户输入，并记录图执行轨迹。",
    "classify": "使用 LangChain 分类链判断 category、priority、action 和 reason。",
    "route": "根据 action 把问题路由到 RAG、人工升级或普通回复。",
    "rag_answer": "调用 Agentic RAG Agent，基于知识库生成带引用的答案。",
    "escalate": "对投诉、安全、严重故障和明确人工诉求生成结构化升级回复。",
    "general_response": "对寒暄或知识库范围外问题给出边界说明，不编造客服政策。",
    "final": "统一输出 category、priority、action、answer、citations 和 graph_trace。",
}


class TriageState(TypedDict, total=False):
    message: str
    graph_trace: list[str]
    decision: TriageDecision
    result: TriageResult


class LangGraphTriageAgent:
    def __init__(
        self,
        store: KnowledgeStore,
        classifier: TriageClassifier | None = None,
        rag_agent: AgenticRagAgent | None = None,
    ) -> None:
        self.store = store
        self.classifier = classifier or TriageClassifier()
        self.rag_agent = rag_agent or AgenticRagAgent(store)
        self.graph = self._build_graph()

    def invoke(self, message: str) -> TriageResult:
        state = self.graph.invoke({"message": message, "graph_trace": []})
        return state["result"]

    def _build_graph(self):
        graph = StateGraph(TriageState)
        graph.add_node("intake", self._intake_node)
        graph.add_node("classify", self._classify_node)
        graph.add_node("route", self._route_node)
        graph.add_node("rag_answer", self._rag_answer_node)
        graph.add_node("escalate", self._escalate_node)
        graph.add_node("general_response", self._general_response_node)
        graph.add_node("final", self._final_node)
        graph.add_edge(START, "intake")
        graph.add_edge("intake", "classify")
        graph.add_edge("classify", "route")
        graph.add_conditional_edges(
            "route",
            self._route_after_decision,
            {"answer": "rag_answer", "escalate": "escalate", "general": "general_response"},
        )
        graph.add_edge("rag_answer", "final")
        graph.add_edge("escalate", "final")
        graph.add_edge("general_response", "final")
        graph.add_edge("final", END)
        return graph.compile()

    def _intake_node(self, state: TriageState) -> TriageState:
        return {"graph_trace": self._trace(state, "intake: 已接收用户问题")}

    def _classify_node(self, state: TriageState) -> TriageState:
        decision = self.classifier.classify(state["message"])
        return {
            "decision": decision,
            "graph_trace": self._trace(
                state,
                f"classify: category={decision.category}, priority={decision.priority}, action={decision.action}",
            ),
        }

    def _route_node(self, state: TriageState) -> TriageState:
        decision = state["decision"]
        return {"graph_trace": self._trace(state, f"route: 选择 {decision.action} 路径，原因：{decision.reason}")}

    def _route_after_decision(self, state: TriageState) -> str:
        return state["decision"].action

    def _rag_answer_node(self, state: TriageState) -> TriageState:
        decision = state["decision"]
        rag_result = self.rag_agent.query(state["message"])
        return {
            "result": TriageResult(
                category=decision.category,
                priority=decision.priority,
                action="answer",
                answer=rag_result.answer,
                citations=rag_result.citations,
                graph_trace=self._trace(state, f"rag_answer: 返回 {len(rag_result.citations)} 条引用"),
            ),
            "graph_trace": self._trace(state, f"rag_answer: 返回 {len(rag_result.citations)} 条引用"),
        }

    def _escalate_node(self, state: TriageState) -> TriageState:
        decision = state["decision"]
        answer = (
            "该问题已进入人工分诊路径。建议提交给支持队列，并附带用户原始描述、发生时间、"
            "影响范围、错误截图或 request_id；如果是投诉、安全或 P0 故障，应按高优先级处理。"
        )
        trace = self._trace(state, "escalate: 已生成人工升级摘要")
        return {
            "result": TriageResult(
                category=decision.category,
                priority=decision.priority,
                action="escalate",
                answer=answer,
                citations=[],
                graph_trace=trace,
            ),
            "graph_trace": trace,
        }

    def _general_response_node(self, state: TriageState) -> TriageState:
        decision = state["decision"]
        answer = "这是普通咨询或当前客服知识库范围外的问题。请补充一个具体的账号、账单、发票、API 或故障类支持问题。"
        trace = self._trace(state, "general_response: 返回范围说明")
        return {
            "result": TriageResult(
                category=decision.category,
                priority=decision.priority,
                action="general",
                answer=answer,
                citations=[],
                graph_trace=trace,
            ),
            "graph_trace": trace,
        }

    def _final_node(self, state: TriageState) -> TriageState:
        trace = self._trace(state, "final: 已生成统一响应")
        result = state["result"]
        return {"result": replace(result, graph_trace=trace), "graph_trace": trace}

    def _trace(self, state: TriageState, event: str) -> list[str]:
        return [*state.get("graph_trace", []), event]


def graph_metadata() -> tuple[str, dict[str, str]]:
    return TRIAGE_GRAPH_MERMAID, TRIAGE_GRAPH_NODES
