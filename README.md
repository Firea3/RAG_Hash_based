# deepseek-agent-rag

> 为 DeepSeek Agent 提供本地知识库检索（RAG）能力的最小可用版本 **v0.1**。
> 零第三方依赖、完全离线可跑、以 **Agent 工具 / Skill** 形式接入：Agent 先检索带来源的原文片段，再据实回答并用 `[1]`、`[2]` 标注引用。

## 1. 为什么做这个项目

DeepSeek Agent 直接回答企业私有文档、项目手册、产品 FAQ 时，容易凭空编造。
本项目把本地文档切成片段、转成向量、保存为 JSON 索引；Agent 通过 `search_knowledge_base` 工具检索最相关的原文，
**答案仍由 Agent 的大模型组织**，但事实来源可追溯、可核验。

## 2. 核心特性

- **文档加载**：递归加载 `.md` / `.markdown` / `.txt` / `.json`，支持目录与单文件。
- **中文友好分块**：段落 → 句子 → 贪心打包，`chunk_size=420`、`chunk_overlap=80` 可配置，相邻片段带重叠。
- **可插拔嵌入**：默认离线确定性 hashing 嵌入（256 维，无需网络/Key）；可切换 OpenAI 兼容 `/embeddings` 接口。
- **轻量向量存储**：纯 Python 余弦相似度 top-k，JSON 持久化，排序确定（`(-score, chunk_id)`）。
- **Agent 工具**：标准 OpenAI / DeepSeek function calling schema，`run_tool()` 接受 JSON 字符串。
- **CLI**：`ingest` / `query` / `tool-schema` 三个子命令，退出码 0 / 1 / 2。
- **零依赖、可测试**：仅标准库，离线 `unittest` 全绿（详见 [docs/05_test_report.md](docs/05_test_report.md)）。
- **逐行中文注释**：`rag/` 全部代码每行可执行语句上方都有面向初学者的中文注释（规范见 [docs/07_comment_guideline.md](docs/07_comment_guideline.md)）。

## 3. 快速开始

要求：Python 3.10+（本机 3.13）。**无需 `pip install` 任何包。**

```bash
# 1) 构建索引（示例知识库在 data/knowledge_base/，索引默认写到 data/index.json）
python -m rag.cli ingest data/knowledge_base

# 2) 检索：默认逐行打印 [序号] 分数 来源
python -m rag.cli query "如何把知识库接入 DeepSeek Agent"

# 3) 输出带 [1] 引用的上下文（可直接拼进答案模板）
python -m rag.cli query "如何把知识库接入 DeepSeek Agent" --context

# 4) 输出 JSON（结构与工具返回一致）
python -m rag.cli query "知识库怎么接入" --json --top-k 2

# 5) 打印 Agent 工具 schema
python -m rag.cli tool-schema
```

> 文档更新后执行 `python -m rag.cli ingest data/knowledge_base --rebuild` 全量重建；v0.1 不支持增量更新。

## 4. 接入 DeepSeek Agent

### 方式 A：工具调用（推荐）

1. 把 `python -m rag.cli tool-schema` 的输出（或 [skills/deepseek-rag/tool_schema.json](skills/deepseek-rag/tool_schema.json)）注册为 Agent 的 function tool；
2. 使用 [prompts/rag_system_prompt.md](prompts/rag_system_prompt.md) 作为系统提示词（先检索、引用 `[n]`、不足即明说、最多检索两次）；
3. 工具执行时把模型返回的 JSON 参数交给 `run_tool`：

```python
from rag import Rag, run_tool

rag = Rag()  # 自动加载 data/index.json；也可传 RagConfig 指向其他索引
result = run_tool('{"query": "如何把知识库接入 DeepSeek Agent", "top_k": 4}', rag=rag)
for item in result["results"]:
    print(item["score"], item["source"])
    print(item["text"])
```

返回结构固定为：

```json
{"query": "...", "results": [{"text": "...", "source": "...", "score": 0.812345, "chunk_id": "..."}]}
```

### 方式 B：预检索 + 答案模板（无工具调用）

```python
from rag import Rag

rag = Rag()
context = rag.context("如何把知识库接入 DeepSeek Agent", top_k=4)  # 形如 [1] 来源：...
prompt = open("prompts/rag_answer_prompt.md", encoding="utf-8").read()
answer_input = prompt.replace("{context}", context).replace("{question}", "如何把知识库接入 DeepSeek Agent？")
```

### 方式 C：Agent Skill

[skills/deepseek-rag/SKILL.md](skills/deepseek-rag/SKILL.md) 提供可直接安装/加载的 Skill，
包含触发条件、工具参数、CLI、引用与兜底规则、故障排查与已知限制；同目录 `tool_schema.json` 与 `rag/tool.py` 的 `TOOL_SPEC` 完全一致。

## 5. 架构与数据流

```
接入层   DeepSeek Agent ── function call ──► search_knowledge_base / Skill
            │                                        │
应用层      CLI (ingest/query/tool-schema)      Rag 门面 (ingest/query/context)
            └──────────────┬─────────────────────────┘
领域层              loaders → chunker → embedder → store
数据层        data/knowledge_base/**        data/index.json
```

- `ingest`：加载文档 → 分块（带重叠）→ 批量嵌入 → 追加/重建 → 写入 JSON 索引。
- `query`：嵌入问题 → 余弦相似度排序 → top-k 命中（含 `source`、`score`、`chunk_id`）。
- `context`：把命中拼成带 `[1] 来源：...` 编号的上下文。

模块职责、ADR 与异常边界见 [docs/03_architecture.md](docs/03_architecture.md)。

## 6. 目录结构

```
project_1/
├── rag/                     # 核心代码（11 个模块，逐行中文注释）
├── tests/                   # 离线 unittest 自动化测试
├── data/knowledge_base/     # 示例知识库（md / txt / json）
├── data/index.json          # 构建出的向量索引（可删除后重新 ingest）
├── prompts/                 # Agent 系统提示词与答案模板
├── skills/deepseek-rag/     # Agent Skill 与工具 schema
└── docs/                    # 00 契约 ~ 07 注释规范（需求/架构/测试/纪要）
```

## 7. 配置与参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `chunk_size` | 420 | 片段最大字符数 |
| `chunk_overlap` | 80 | 相邻片段重叠字符数，需 `< chunk_size` |
| `top_k` | 4 | 默认返回片段数（工具限制 1~10） |
| `store_path` | `data/index.json` | 索引文件路径 |
| `embedder` | `hashing` | `hashing`（离线）或 `openai`（OpenAI 兼容接口） |
| `dim` | 256 | hashing 维度；切换 openai 时显式指定（如 1024） |
| `base_url` | `https://api.deepseek.com` | OpenAI 兼容接口地址 |
| `embed_model` | `deepseek-embedding` | 嵌入模型名 |

切换到 API 嵌入（需要网络与 Key，**DeepSeek 官方接口是否提供 embeddings 以实际为准**，可指向任意 OpenAI 兼容服务）：

```bash
$env:RAG_API_KEY = "your-key"   # PowerShell；CMD 用 set，bash 用 export
python -m rag.cli ingest data/knowledge_base --rebuild --embedder openai --dim 1024 --base-url https://api.deepseek.com --embed-model deepseek-embedding
```

## 8. 索引格式（version=1）

```json
{
  "version": 1,
  "embedder": {"name": "hashing", "dim": 256},
  "chunks": [
    {"chunk_id": "data/knowledge_base/agent_faq.md#0", "doc_id": "data/knowledge_base/agent_faq.md",
     "text": "...", "source": "data/knowledge_base/agent_faq.md", "metadata": {"chunk_index": 0}}
  ],
  "vectors": [[0.0, 0.0, "..."]]
}
```

索引为 UTF-8 JSON，可直接查看、备份、迁移；维度不一致时会提示删除索引后重新 `ingest`。

## 9. 测试与验收

```bash
python -m unittest discover -s tests -t . -v       # 93 条用例，全部通过，约 0.54 秒
python -m compileall rag                            # 语法检查
```

| 验收项 | 状态 |
| --- | --- |
| AC1 自动化测试全绿 | ✅ 93 条用例 100% 通过（0.54s、离线、0 第三方依赖），见 [docs/05_test_report.md](docs/05_test_report.md) |
| AC2 示例库 ingest + query + context 引用 | ✅ 冒烟通过 |
| AC3 工具参数校验与 `results` 结构 | ✅ |
| AC4 CLI 子命令与退出码 0/1/2 | ✅ |
| AC5 无网络、无 API Key 全链路可跑 | ✅ |
| AC6 零第三方依赖 | ✅（AST 扫描仅标准库 + `rag`） |
| AC7 文档、prompt、Skill 齐全 | ✅ |

## 10. 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/00_project_brief.md](docs/00_project_brief.md) | **权威契约**：需求确认、范围、验收标准、接口签名、索引格式、工具协议 |
| [docs/01_requirements.md](docs/01_requirements.md) | 需求文档：用户故事、FR/NFR、范围外、AC、需求追踪矩阵 |
| [docs/02_project_plan.md](docs/02_project_plan.md) | 项目计划：里程碑、分工、依赖图、DoD、v0.2 backlog |
| [docs/03_architecture.md](docs/03_architecture.md) | 架构设计：分层、数据流、ADR、异常边界、容量与演进 |
| [docs/04_test_plan.md](docs/04_test_plan.md) | 测试方案：策略、90 条用例设计、覆盖矩阵、准入门槛 |
| [docs/05_test_report.md](docs/05_test_report.md) | 测试报告：执行结果、缺陷与修复、结论 |
| [docs/06_meeting_minutes.md](docs/06_meeting_minutes.md) | 会议室讨论纪要与决议（产品/架构/代码/注释/测试/AI） |
| [docs/07_comment_guideline.md](docs/07_comment_guideline.md) | 逐行中文注释规范与自检方法 |

## 11. 已知限制与 v0.2 方向

**v0.1 明确不做**：PDF/OCR、增量索引与文件监听、rerank、多租户/权限、HTTP 服务、Web/GUI、答案生成（由 Agent 大模型完成）。

已知限制：

- hashing 嵌入是字符 n-gram 特征哈希，适合演示与离线测试，**语义召回弱于真实嵌入模型**；追求效果请切换 OpenAI 兼容嵌入。
- 纯 Python 余弦在万级片段内可接受；每次工具调用 `Rag()` 会重新加载 JSON 索引，超大索引需服务化/缓存。
- 仅 `.md` / `.markdown` / `.txt` / `.json`；JSON 顶层需为数组或 `{"items": [...]}`。
- 片段重叠按字符截取：相邻片段开头可能是上一片段的半句（例如“装与依赖”），这是保证答案不被切断的代价，不影响检索与引用；追求更整齐可把 `chunk_overlap` 调小（如 0~40）。

v0.2 backlog：PDF 解析、增量索引、真实嵌入联调、rerank / 混合检索、HTTP 检索服务、评测集与召回指标。
