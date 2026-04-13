# Support Copilot 模块说明

本文档说明当前项目每个模块的职责，以及本次双 Agent 重构后实际保留和实现的内容。

## 1. 项目定位

当前项目是一个后端 API 作品项目，只保留两个核心 Agent：

- Agentic RAG Agent：负责从客服知识库检索、过滤、引用来源并生成 grounded answer。
- LangGraph Triage Agent：负责客服问题分诊，并在知识类问题上调用 Agentic RAG。

项目已经删除前端、登录认证、会话轮询、Celery、Redis runtime cache、旧流程图图片和旧全栈规划文档，避免偏离“两个 Agent”的展示重点。

## 2. 根目录文件

### `README.md`

- 项目主说明文档。
- 说明当前项目只展示 Agentic RAG 和 LangGraph Triage Agent。
- 包含 LangGraph 分诊流程 Mermaid 图、节点说明、Docker 启动命令、demo 知识库导入命令、Agent API 调用命令、RAGAS 评估命令和简历表述。

### `docker-compose.yml`

- 定义本地一键运行环境。
- 启动两个服务：`api` 和 `postgres`。
- `api` 使用 `backend/Dockerfile` 构建 FastAPI 后端。
- `postgres` 使用 `pgvector/pgvector:pg16` 镜像，启动时挂载 `db/001_init_pgvector.sql` 初始化 pgvector 和知识库表。
- 不再启动 Redis、Celery worker 或前端服务。

### `.env.example`

- 环境变量模板，不包含真实密钥。
- 包含数据库连接、LLM provider、模型名、API key 占位、base URL、是否启用真实模型调用、RAG 最低相关度阈值、上下文长度、RAGAS 禁用遥测开关。

### `.env`

- 本地实际环境变量文件。
- 可能包含真实 API key，只在本机使用，不应提交或复制到文档、日志、截图中。
- 本次只保留双 Agent 后端和 RAGAS 需要的配置项。

### `.gitignore`

- 忽略本地环境变量、虚拟环境、Python 缓存、pytest 缓存、构建产物、临时 TypeScript 文件和日志目录。

### `.dockerignore`

- 排除 Docker 构建上下文中不需要的本地文件。
- 避免把 `.env`、虚拟环境、缓存等内容打进镜像。

## 3. 后端应用模块

后端源码位于 `backend/app/`。

### `backend/app/main.py`

- FastAPI 应用入口。
- 初始化 `PgVectorStore`，并把它注入到 API 路由中。
- 已实现接口：
  - `GET /health`
  - `POST /knowledge/ingest-demo`
  - `POST /agents/rag/query`
  - `POST /agents/triage/invoke`
  - `GET /agents/triage/graph`
- 不再提供认证、注册、登录、会话、任务状态轮询或前端相关接口。

### `backend/app/config.py`

- 统一读取环境变量。
- 当前配置项包括：应用名、运行环境、数据库 URL、LLM provider、聊天模型、embedding 模型名、API key、base URL、是否启用真实 LLM 调用、RAG 最低分数阈值、RAG 最大上下文字符数。
- 使用 `pydantic-settings` 读取 `.env`。
- 不写死真实密钥。

### `backend/app/models.py`

- 定义 Agent 和知识库使用的内部数据结构。
- 当前实体包括：
  - `KnowledgeDocument`
  - `SearchResult`
  - `Citation`
  - `RagResult`
  - `TriageDecision`
  - `TriageResult`
- 已删除旧版用户、会话、消息、任务、长期记忆等全栈 MVP 模型。

### `backend/app/schemas.py`

- 定义 FastAPI 请求和响应 schema。
- 当前 schema 包括：
  - `HealthResponse`
  - `KnowledgeIngestResponse`
  - `CitationRead`
  - `RagQueryRequest`
  - `RagQueryResponse`
  - `TriageInvokeRequest`
  - `TriageInvokeResponse`
  - `TriageGraphResponse`
- 对输入长度和 `top_k` 做基础约束。

### `backend/app/postgres_store.py`

- 数据存储和检索层。
- `KnowledgeStore` 定义 Agent 需要的最小数据访问协议。
- `PgVectorStore` 使用 PostgreSQL/pgvector 持久化 demo 知识库和 chunk，并进行向量检索。
- `InMemoryKnowledgeStore` 供最小单元场景使用，不依赖数据库。
- 包含文本分块、确定性文本向量 fallback、中文/英文 token 切分、相关度评分等基础逻辑。
- 当前检索不做用户隔离，因为本项目已经删除登录和多用户会话能力。

### `backend/app/llm_router.py`

- LangChain 模型和提示词链封装。
- `build_chat_model()` 根据 `.env` 创建 OpenAI-compatible `ChatOpenAI`。
- `RagAnswerComposer` 使用 LangChain Runnable 生成 RAG 答案。
- `TriageClassifier` 使用 LangChain Runnable 输出分诊决策。
- `LLM_ENABLE_CALLS=false` 时使用本地 fallback，不外呼模型。
- 真实模型模式下通过 `ChatPromptTemplate | ChatOpenAI | StrOutputParser/JsonOutputParser` 调用模型。
- 已修复旧版 Agent 中文提示词和 fallback 文案乱码。

### `backend/app/__init__.py`

- Python package 标记文件。
- 让 `app.*` 模块可以被 FastAPI、评估脚本和测试导入。

## 4. Agent 模块

Agent 代码位于 `backend/app/agent/`。

### `backend/app/agent/rag.py`

- 实现 `AgenticRagAgent`。
- 主流程：检索知识库、按最低相关度过滤、无命中时返回 no-answer、命中时调用 `RagAnswerComposer` 生成回答。
- 输出 `RagResult`，包含：
  - `answer`
  - `citations`
  - `confidence`
  - `no_answer_reason`
- citations 会携带文档标题、source、摘要片段和分数。

### `backend/app/agent/triage.py`

- 实现 `LangGraphTriageAgent`。
- 使用 LangGraph `StateGraph` 构建固定流程：
  `intake -> classify -> route -> rag_answer/escalate/general_response -> final`。
- `classify` 节点调用 `TriageClassifier`。
- `rag_answer` 节点调用 `AgenticRagAgent`。
- `escalate` 节点输出人工升级建议。
- `general_response` 节点输出范围说明，避免把非客服知识库问题硬转成 RAG 回答。
- `graph_metadata()` 返回 Mermaid 图和节点说明，供 API 和 README 展示。

### `backend/app/agent/__init__.py`

- 导出 `AgenticRagAgent` 和 `LangGraphTriageAgent`。
- 方便其他模块通过 `app.agent` 统一导入核心 Agent。

## 5. Demo 知识库模块

### `backend/app/data/__init__.py`

- Python package 标记文件。

### `backend/app/data/demo_knowledge.py`

- 内置客服知识库样例。
- 当前包含账号密码、退款账单、发票、数据导出、账号锁定、故障 SLA、取消订阅、隐私删除、API 429 限流等文档。
- 每条文档包含：
  - `slug`
  - `title`
  - `source`
  - `content`
- `POST /knowledge/ingest-demo` 和 `backend/eval/run_eval.py` 都会使用这份数据。

## 6. 数据库模块

### `db/001_init_pgvector.sql`

- 初始化 PostgreSQL/pgvector。
- 启用 `vector` extension。
- 创建 `knowledge_documents` 表保存知识库文档。
- 创建 `knowledge_chunks` 表保存切分后的知识片段和 1536 维向量。
- 创建 `knowledge_documents.slug` 唯一索引，支持 demo 文档重复导入时更新。
- 创建 `knowledge_chunks.embedding` HNSW cosine 向量索引。
- 包含兼容旧表的 `alter table` 语句，避免已有旧 volume 初始化失败。

## 7. 评估模块

评估代码位于 `backend/eval/`。

### `backend/eval/eval_cases.json`

- RAGAS 评估用例数据。
- 每条样本包含：
  - `question`
  - `expected_action`
  - `expected_source`
  - `reference`
- RAG 样本会进入 RAGAS 数据集；人工升级和普通寒暄样本会被跳过，因为它们不适合用 RAG 指标评价。

### `backend/eval/run_eval.py`

- 使用 RAGAS 执行轻量离线 RAG 评估。
- 启动时默认设置 `RAGAS_DO_NOT_TRACK=true`，避免 RAGAS 匿名统计。
- 流程：
  1. 读取 `.env`。
  2. 创建评审 LLM。
  3. 导入内置 demo 知识库。
  4. 调用 LangGraph 分诊 Agent 收集 RAG 样本。
  5. 用 `EvaluationDataset.from_list()` 构建 RAGAS 数据集。
  6. 用 `ragas.evaluate()` 运行指标。
  7. 输出 RAGAS 分数、样本数量、跳过的非 RAG 样本和平均 Agent 延迟。
- 当前指标包括 `LLMContextRecall`、`Faithfulness`、`FactualCorrectness`，并在当前 RAGAS 版本支持时加入 `LLMContextPrecisionWithReference`。

## 8. 测试模块

### `backend/tests/test_agents.py`

- 保留最小后端单元场景。
- 覆盖：
  - RAG 命中时返回引用。
  - RAG 无命中时不编造答案。
  - 分诊 Agent 将知识库问题路由到 RAG。
  - 分诊 Agent 将投诉/人工诉求路由到人工升级。
- 本文件只作为后端最小行为保护，不包含前端测试。

### `backend/pytest.ini`

- pytest 配置文件。
- 指定 `pythonpath = .`、测试目录 `tests` 和 asyncio 模式。

## 9. Docker 与依赖模块

### `backend/Dockerfile`

- 构建 FastAPI 后端镜像。
- 基于 `python:3.12-slim`。
- 安装 `backend/requirements.txt`。
- 设置 `PYTHONPATH=/app/backend`。
- 默认启动命令为 `uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir /app/backend`。

### `backend/requirements.txt`

- 后端依赖清单。
- 当前包含 FastAPI、Uvicorn、psycopg、Pydantic Settings、LangChain、LangChain OpenAI、LangGraph、RAGAS、pytest 等。
- 已删除旧版 Celery 和 Redis 依赖。

## 10. 已删除的旧模块

这些内容已经不属于当前双 Agent 项目：

- `frontend/`：旧 React 前端。
- `logs/`：旧本地运行日志。
- `docs/` 旧版文档：旧全栈 MVP 模块说明和运行说明已经替换为当前两份新文档。
- `Support Copilot.md`：旧全栈规划文档。
- `whiteboard_exported_image.png`：旧流程图图片。
- `backend/app/auth.py`：旧 JWT 登录认证。
- `backend/app/state_cache.py`：旧 Redis runtime cache。
- `backend/app/tasks.py`：旧 Celery 任务。
- `backend/app/agent/tools.py` 和 `backend/app/agent/workflow.py`：旧会话型 Agent workflow，已替换为当前 `rag.py` 和 `triage.py`。
