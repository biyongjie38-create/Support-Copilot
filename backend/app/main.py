from dataclasses import asdict
from typing import Annotated

from fastapi import Depends, FastAPI, status

from app.agent.rag import AgenticRagAgent
from app.agent.triage import LangGraphTriageAgent, graph_metadata
from app.config import get_settings
from app.data.demo_knowledge import DEMO_KNOWLEDGE
from app.postgres_store import KnowledgeStore, PgVectorStore
from app.schemas import (
    HealthResponse,
    KnowledgeIngestResponse,
    RagQueryRequest,
    RagQueryResponse,
    TriageGraphResponse,
    TriageInvokeRequest,
    TriageInvokeResponse,
)


def create_app(store: KnowledgeStore | None = None) -> FastAPI:
    settings = get_settings()
    knowledge_store = store or PgVectorStore(settings.database_url)

    app = FastAPI(
        title=settings.app_name,
        description="Backend-only portfolio API for an Agentic RAG agent and a LangGraph triage agent.",
    )
    app.state.settings = settings
    app.state.store = knowledge_store

    def current_store() -> KnowledgeStore:
        return app.state.store

    StoreDep = Annotated[KnowledgeStore, Depends(current_store)]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", app=settings.app_name)

    @app.post(
        "/knowledge/ingest-demo",
        response_model=KnowledgeIngestResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def ingest_demo(store: StoreDep) -> KnowledgeIngestResponse:
        documents = store.ingest_documents(DEMO_KNOWLEDGE)
        return KnowledgeIngestResponse(
            inserted_documents=len(documents),
            document_titles=[document.title for document in documents],
        )

    @app.post("/agents/rag/query", response_model=RagQueryResponse)
    def query_rag(payload: RagQueryRequest, store: StoreDep) -> dict:
        result = AgenticRagAgent(store).query(payload.question, top_k=payload.top_k)
        return asdict(result)

    @app.post("/agents/triage/invoke", response_model=TriageInvokeResponse)
    def invoke_triage(payload: TriageInvokeRequest, store: StoreDep) -> dict:
        result = LangGraphTriageAgent(store).invoke(payload.message)
        return asdict(result)

    @app.get("/agents/triage/graph", response_model=TriageGraphResponse)
    def triage_graph() -> TriageGraphResponse:
        mermaid, nodes = graph_metadata()
        return TriageGraphResponse(mermaid=mermaid, nodes=nodes)

    return app


app = create_app()
