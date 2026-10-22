"""Agent 工具：function calling 协议定义与执行入口。"""

# 需要 json 模块解析字符串参数：Agent 传来的 arguments 通常是一段 JSON 文本。
import json
# 需要 Mapping 类型：表示“键值对”接口，字典就是最常见的 Mapping。
from collections.abc import Mapping
# 需要 Any 类型注解：参数里的值类型可能多种多样，用 Any 表示“任意类型”。
from typing import Any

# 需要 Rag 管道：run_tool 真正检索时会调用它的 query 方法。
from rag.pipeline import Rag

# 定义 Agent 工具说明书：用 function calling 格式告诉大模型工具名、用途和参数，像贴在工具箱外的使用说明。
TOOL_SPEC: dict = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "在本地知识库中检索与问题最相关的原文片段，返回带来源的文本，供 Agent 据实回答并标注引用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题，建议改写成关键词或完整问句。"},
                "top_k": {
                    "type": "integer",
                    "description": "返回片段数量，取值范围 1 到 10，默认 4。",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
}


# 定义 run_tool：执行检索工具，接收 JSON 字符串或参数字典，返回含 query 与 results 的字典。
def run_tool(arguments: Mapping[str, Any] | str, rag: Rag | None = None) -> dict[str, Any]:
    """执行检索工具。参数：arguments JSON 字符串或参数字典、rag 可复用的管道；返回：含 query 与 results 的字典。"""
    # 如果参数是字符串，说明是 Agent 传来的 JSON 文本，需要先解析。
    if isinstance(arguments, str):
        # 尝试把 JSON 文本解析成字典或列表：格式不对会抛 JSONDecodeError。
        try:
            # 解析成功就把 arguments 换成 Python 对象：例如 '{"query": "怎么接入"}' -> {"query": "怎么接入"}。
            arguments = json.loads(arguments)
        # 接住 JSON 语法错误，换成带行号的中文 ValueError，方便定位。
        except json.JSONDecodeError as exc:
            # 报错并说明第几行出错：像老师说“第 1 行引号没配对”。
            raise ValueError(f"参数不是合法 JSON：{exc.msg}（第 {exc.lineno} 行）") from exc
    # 解析后必须是“键值对”结构（Mapping），列表或数字都不行。
    if not isinstance(arguments, Mapping):
        # 报错并给出正确示例：像填表时提示“请填成 {query: 问题, top_k: 4} 的样子”。
        raise ValueError("参数必须是 JSON 对象，例如 {\"query\": \"问题\", \"top_k\": 4}")
    # 取出 query 参数：没有这个键时得到 None。
    query = arguments.get("query")
    # query 必须是非空字符串：None、数字、空串都不合格。
    if not isinstance(query, str) or not query.strip():
        # 报错提示 query 的要求：像报名表上“姓名”一栏不能空着。
        raise ValueError("query 必须是非空字符串")
    # 取出 top_k 参数，用户没传就默认 4。
    top_k = arguments.get("top_k", 4)
    # top_k 必须是 1~10 的整数；bool 是 int 的子类，所以要单独排除 True/False。
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
        # 报错并显示实际收到的值：像调音量只能 1~10，收到 99 就提示。
        raise ValueError(f"top_k 必须是 1 到 10 之间的整数，当前为 {top_k!r}")
    # 用传入的管道检索；没传就临时新建一个默认 Rag()。
    hits = (rag or Rag()).query(query, top_k)
    # 把每个 Hit 转成字典（正文、来源、保留 6 位小数的分数、片段编号），组成 results 列表。
    results = [
        {"text": hit.text, "source": hit.source, "score": round(hit.score, 6), "chunk_id": hit.chunk_id}
        for hit in hits
    ]
    # 按契约返回固定结构：{"query": 原问题, "results": [...]}。
    return {"query": query, "results": results}
