from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any


os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.agent.triage import LangGraphTriageAgent
from app.config import get_settings
from app.data.demo_knowledge import DEMO_KNOWLEDGE
from app.llm_router import build_chat_model
from app.postgres_store import PgVectorStore


def load_cases() -> list[dict[str, Any]]:
    return json.loads(Path(__file__).with_name("eval_cases.json").read_text(encoding="utf-8"))


def build_ragas_metrics() -> list[Any]:
    from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall

    metrics: list[Any] = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
    try:
        from ragas.metrics import LLMContextPrecisionWithReference

        metrics.insert(0, LLMContextPrecisionWithReference())
    except ImportError:
        pass
    return metrics


def serialize_ragas_result(result: Any) -> dict[str, Any]:
    try:
        return dict(result)
    except (TypeError, ValueError):
        pass
    try:
        return {"scores": result.to_pandas().to_dict(orient="records")}
    except Exception:
        return {"raw": str(result)}


def main() -> None:
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper

    settings = get_settings()
    evaluator_llm = build_chat_model()
    if evaluator_llm is None:
        raise SystemExit(
            "RAGAS 评估需要评审模型。请在 .env 设置 LLM_ENABLE_CALLS=true，并配置 LLM_API_KEY、"
            "LLM_BASE_URL、LLM_CHAT_MODEL。"
        )

    store = PgVectorStore(settings.database_url)
    store.ingest_documents(DEMO_KNOWLEDGE)
    triage_agent = LangGraphTriageAgent(store)

    ragas_samples: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, str]] = []
    total_latency_ms = 0.0

    for case in load_cases():
        start = time.perf_counter()
        result = triage_agent.invoke(case["question"])
        total_latency_ms += (time.perf_counter() - start) * 1000

        if case["expected_action"] != "answer":
            skipped_cases.append({"question": case["question"], "reason": "non_rag_case"})
            continue

        retrieved_contexts = [
            document.content
            for document in store.search_knowledge(case["question"], limit=case.get("top_k", 5))
            if document.score >= settings.rag_min_score
        ]
        ragas_samples.append(
            {
                "user_input": case["question"],
                "retrieved_contexts": retrieved_contexts,
                "response": result.answer,
                "reference": case["reference"],
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(ragas_samples)
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=build_ragas_metrics(),
        llm=LangchainLLMWrapper(evaluator_llm),
    )

    report = {
        "evaluator": "ragas",
        "ragas_scores": serialize_ragas_result(ragas_result),
        "ragas_sample_count": len(ragas_samples),
        "skipped_non_rag_cases": skipped_cases,
        "average_agent_latency_ms": round(total_latency_ms / max(len(ragas_samples) + len(skipped_cases), 1), 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
