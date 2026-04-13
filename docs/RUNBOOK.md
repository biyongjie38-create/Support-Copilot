# Support Copilot 实际运行流程

本文档说明当前双 Agent 版本如何在本地实际运行。项目没有前端页面，所有演示都通过后端 API、Docker、PowerShell 命令和 RAGAS 评估脚本完成。

## 1. 前置条件

本机需要：

- Docker Desktop
- Python 3.11+ 或 3.12
- PowerShell

进入项目根目录：

```powershell
Set-Location "D:\Agent\Support Copilot"
```

## 2. 环境变量

首次运行复制模板：

```powershell
Copy-Item .env.example .env
```

`.env` 中至少确认：

```text
DATABASE_URL=postgresql://support:support@localhost:5432/support_copilot
LLM_PROVIDER=ollama
LLM_CHAT_MODEL=qwen2.5:7b
LLM_API_KEY=
LLM_BASE_URL=http://localhost:11434/v1
LLM_ENABLE_CALLS=false
RAG_MIN_SCORE=0.18
RAG_MAX_CONTEXT_CHARS=5000
RAGAS_DO_NOT_TRACK=true
```

如果要接通义千问 OpenAI-compatible 接口，可以改为：

```text
LLM_PROVIDER=qwen
LLM_CHAT_MODEL=qwen3.5-flash
LLM_API_KEY=你的 API Key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_ENABLE_CALLS=true
RAGAS_DO_NOT_TRACK=true
```

不要把真实 `LLM_API_KEY` 写入 README、截图、日志或提交记录。

## 3. Docker 一键启动

启动 API 和 Postgres/pgvector：

```powershell
docker compose up --build --remove-orphans
```

服务说明：

- API 地址：`http://127.0.0.1:8000`
- Postgres 地址：`localhost:5432`
- 数据库名：`support_copilot`
- 数据库用户：`support`

如果 5432 或 8000 端口被占用，先停掉占用旧容器或改 `docker-compose.yml` 端口映射。

## 4. 初始化数据库

新 volume 首次启动时会自动执行：

```text
db/001_init_pgvector.sql
```

如果你复用了旧 volume，手动补跑一次：

```powershell
docker compose exec -T postgres psql -U support -d support_copilot -f /docker-entrypoint-initdb.d/001_init_pgvector.sql
```

## 5. 健康检查

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

预期返回类似：

```json
{
  "status": "ok",
  "app": "Support Copilot"
}
```

## 6. 导入内置客服知识库

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/knowledge/ingest-demo"
```

该接口会把 `backend/app/data/demo_knowledge.py` 中的客服文档写入 Postgres/pgvector。重复执行会按 `slug` 更新，不会依赖前端操作。

## 7. 调用 Agentic RAG Agent

请求：

```powershell
$body = @{
  question = "API 一直返回 429 是什么意思？"
  top_k = 5
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/agents/rag/query" `
  -ContentType "application/json" `
  -Body $body
```

响应重点字段：

- `answer`：基于知识库的回答。
- `citations`：引用来源列表，包含文档标题、source、片段和分数。
- `confidence`：根据检索结果计算的置信度。
- `no_answer_reason`：无相关知识时返回原因；有答案时为 `null`。

如果问题不在知识库覆盖范围内，RAG Agent 应明确回答“知识库未覆盖该问题”，而不是猜测。

## 8. 调用 LangGraph 分诊 Agent

请求：

```powershell
$body = @{
  message = "我忘记密码了，应该怎么重置？"
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/agents/triage/invoke" `
  -ContentType "application/json" `
  -Body $body
```

响应重点字段：

- `category`：问题类别。
- `priority`：优先级。
- `action`：`answer`、`escalate` 或 `general`。
- `answer`：最终回复。
- `citations`：如果调用 RAG，会带引用；人工升级和普通回复通常为空。
- `graph_trace`：LangGraph 每个节点的执行轨迹。

分诊逻辑：

- 知识库类问题：进入 `rag_answer`，调用 Agentic RAG。
- 投诉、安全、严重故障、明确转人工：进入 `escalate`。
- 寒暄或客服范围外问题：进入 `general_response`。

## 9. 查看分诊流程图

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/agents/triage/graph"
```

该接口返回：

- Mermaid 图文本。
- 每个节点的中文说明。

可以把返回内容直接放进 README、简历项目说明或面试演示材料。

## 10. 运行 RAGAS 评估

RAGAS 评估会调用评审模型。先确认 `.env`：

```text
LLM_ENABLE_CALLS=true
LLM_API_KEY=你的 API Key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_CHAT_MODEL=qwen3.5-flash
RAGAS_DO_NOT_TRACK=true
```

然后执行：

```powershell
python backend\eval\run_eval.py
```

脚本流程：

1. 导入内置 demo 知识库。
2. 读取 `backend/eval/eval_cases.json`。
3. 调用 LangGraph 分诊 Agent。
4. 只把 `expected_action=answer` 的 RAG 样本送入 RAGAS。
5. 使用 `EvaluationDataset.from_list()` 构建 RAGAS 数据集。
6. 使用 `ragas.evaluate()` 输出指标。

当前 RAGAS 指标：

- `context_precision`
- `context_recall`
- `faithfulness`
- `factual_correctness`

人工升级和普通寒暄样本会被记录到 `skipped_non_rag_cases`，不会进入 RAGAS RAG 指标。

## 11. 本地非 Docker 运行

如果不使用 Docker 跑 API，只想本地启动 FastAPI：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn app.main:app --reload --app-dir backend
```

仍然需要单独启动 Postgres/pgvector：

```powershell
docker compose up postgres
```

## 12. 常见问题

### 端口被占用

查看端口：

```powershell
netstat -ano | Select-String -Pattern ':8000|:5432'
```

如果是旧 compose 容器占用，可以先停止旧容器，再重新启动当前项目：

```powershell
docker compose down --remove-orphans
docker compose up --build --remove-orphans
```

### RAGAS 提示没有评审模型

说明 `.env` 里还没有打开真实模型调用。确认：

```text
LLM_ENABLE_CALLS=true
LLM_API_KEY=你的 API Key
LLM_BASE_URL=你的 OpenAI-compatible 地址
LLM_CHAT_MODEL=你的模型名
```

### RAG 没有引用

先确认已经导入 demo 知识库：

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/knowledge/ingest-demo"
```

如果问题确实不在知识库范围内，返回“知识库未覆盖该问题”是预期行为。
