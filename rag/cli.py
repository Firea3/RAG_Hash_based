"""命令行接口：ingest / query / tool-schema。"""

# 需要 argparse 解析命令行参数：像收件窗口按表格检查你要办哪项业务。
import argparse
# 需要 json 打印工具说明书和 JSON 结果：ensure_ascii=False 可以让中文原样输出。
import json
# 需要 sys 把错误打印到标准错误：保证终端里正常输出和错误输出不混在一起。
import sys
# 需要 Sequence 类型注解：表示命令行参数是一串字符串。
from collections.abc import Sequence

# 需要 RagConfig 配置类：把命令行参数整理成配置对象。
from rag.config import RagConfig
# 需要 Rag 管道：ingest 和 query 都靠它执行。
from rag.pipeline import Rag
# 需要 TOOL_SPEC 工具说明书和 run_tool 执行函数：分别用于 tool-schema 和 --json 输出。
from rag.tool import TOOL_SPEC, run_tool


# 定义 main：命令行总入口，负责分发子命令并返回退出码。
def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。参数：argv 参数序列（默认取 sys.argv）；返回：退出码，成功 0、运行错误 1。"""
    # 解析命令行参数：argparse 会处理 --help、参数缺失等用法错误（退出码 2）。
    args = _build_parser().parse_args(argv)
    # 如果是 tool-schema 子命令，直接打印 JSON 后结束。
    if args.command == "tool-schema":
        # 打印工具说明书：indent=2 便于人读，ensure_ascii=False 让中文不转义。
        print(json.dumps(TOOL_SPEC, ensure_ascii=False, indent=2))
        # 返回 0 表示成功：像考试交卷并且通过了。
        return 0
    # 其他子命令可能出错（文件不存在、JSON 非法、网络失败），用 try 包住。
    try:
        # 交给 _run 执行具体子命令，并返回它的退出码。
        return _run(args)
    # 接住三类运行期错误：文件/网络 IO 错误、参数不合法、运行环境错误。
    except (OSError, ValueError, RuntimeError) as exc:
        # 把中文错误信息打印到标准错误：像把问题写到留言板上，不混进正常输出。
        print(f"错误：{exc}", file=sys.stderr)
        # 返回 1 表示运行出错：与 argparse 的用法错误退出码 2 区分开。
        return 1


# 定义 _run：根据子命令执行 ingest 或 query。
def _run(args: argparse.Namespace) -> int:
    # 把命令行参数整理成配置对象：没指定的项用默认值补齐，例如 store_path=data/index.json。
    config = _config_from_args(args)
    # 创建 RAG 管道：会自动加载已有索引，或新建一个空库。
    rag = Rag(config)
    # 如果是 ingest 子命令，就加载文档并建索引。
    if args.command == "ingest":
        # 把路径下的文档灌入索引，rebuild 为真时先清空旧索引。
        summary = rag.ingest(args.paths, rebuild=args.rebuild)
        # 打印入库统计：例如“已入库：3 篇文档，12 个片段 -> data/index.json”。
        print(f"已入库：{summary['documents']} 篇文档，{summary['chunks']} 个片段 -> {config.store_path}")
        # 返回 0 表示成功。
        return 0
    # 如果用户要 JSON 输出（--json），走工具函数生成同样结构。
    if args.as_json:
        # top_k 传了就用命令行值，没传就用配置默认值：例如 4。
        top_k = args.top_k if args.top_k is not None else config.top_k
        # 调用 run_tool 并打印 JSON：结构和 Agent 工具返回完全一致。
        print(json.dumps(run_tool({"query": args.question, "top_k": top_k}, rag=rag), ensure_ascii=False))
        # 返回 0 表示成功。
        return 0
    # 如果用户要带引用编号的上下文（--context），直接打印文本。
    if args.context:
        # 生成带 [n] 编号的上下文：每个命中一块。
        text = rag.context(args.question, args.top_k)
        # 有内容才打印：没有命中就什么都不输出，保持安静。
        if text:
            # 打印上下文文本。
            print(text)
        # 返回 0 表示成功。
        return 0
    # 默认检索模式：查询命中的片段。
    hits = rag.query(args.question, args.top_k)
    # 没有命中时打印友好提示。
    if not hits:
        # 告诉用户没找到：像查字典没查到这个词。
        print("未找到相关内容。")
        # 返回 0：没有命中不算错误。
        return 0
    # 逐条打印命中结果，序号从 1 开始。
    for index, hit in enumerate(hits, start=1):
        # 打印格式：[序号] 分数 来源：路径，例如“[1] 0.812345 来源：data/kb/a.md”。
        print(f"[{index}] {hit.score:.6f} 来源：{hit.source}")
    # 返回 0 表示成功。
    return 0


# 定义 _build_parser：搭建 argparse 解析器，声明所有子命令和参数。
def _build_parser() -> argparse.ArgumentParser:
    # 创建主解析器：prog 显示在帮助里，description 是整体说明。
    parser = argparse.ArgumentParser(prog="python -m rag", description="DeepSeek Agent 本地知识库 RAG 命令行工具。")
    # 添加子命令容器：dest=command 记录用户选了哪个子命令，required=True 表示必须选一个。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 添加 ingest 子命令：负责把文档灌入索引。
    ingest = subparsers.add_parser("ingest", help="把 Markdown / TXT / JSON 文档灌入索引。")
    # 添加路径参数：nargs="+" 表示至少一个路径，也可以一次给多个。
    ingest.add_argument("paths", nargs="+", help="文件或目录路径。")
    # 添加 --rebuild 开关：出现这个参数就为 True，表示先清空旧索引。
    ingest.add_argument("--rebuild", action="store_true", help="先清空现有索引再入库。")
    # 给 ingest 加上公共的存储/嵌入参数（--store、--embedder 等）。
    _add_common_args(ingest)

    # 添加 query 子命令：负责检索知识库。
    query = subparsers.add_parser("query", help="检索知识库。")
    # 添加问题参数：一个位置参数，例如 "知识库怎么接入"。
    query.add_argument("question", help="检索问题。")
    # 添加 --top-k 参数：类型是整数，默认 None（表示用配置里的默认值）。
    query.add_argument("--top-k", type=int, default=None, help="返回片段数量。")
    # 添加 --json 开关：dest=as_json 表示存到 args.as_json，方便后面判断。
    query.add_argument("--json", action="store_true", dest="as_json", help="以 JSON 输出结果。")
    # 添加 --context 开关：打开后输出带 [n] 编号的上下文。
    query.add_argument("--context", action="store_true", help="输出带引用编号的上下文。")
    # 给 query 也加上公共的存储/嵌入参数。
    _add_common_args(query)

    # 添加 tool-schema 子命令：只打印工具说明书，不需要额外参数。
    subparsers.add_parser("tool-schema", help="打印 Agent 工具 schema。")
    # 返回搭好的解析器：main 里调用它的 parse_args 来解析。
    return parser


# 定义 _add_common_args：给 ingest 和 query 共用的参数统一添加，避免重复写。
def _add_common_args(parser: argparse.ArgumentParser) -> None:
    # 添加 --store：指定索引文件路径；默认 None 表示用配置里的 data/index.json。
    parser.add_argument("--store", default=None, help="索引文件路径，默认 data/index.json。")
    # 添加 --embedder：只能选 hashing 或 openai；choices 会自动检查并给出用法错误。
    parser.add_argument("--embedder", choices=["hashing", "openai"], default=None, help="嵌入方式，默认 hashing。")
    # 添加 --dim：向量维度，整数；默认 None 表示用配置里的 256。
    parser.add_argument("--dim", type=int, default=None, help="向量维度。")
    # 添加 --base-url：OpenAI 兼容接口地址；默认 None 表示用配置里的地址。
    parser.add_argument("--base-url", default=None, help="OpenAI 兼容接口地址。")
    # 添加 --embed-model：嵌入模型名；默认 None 表示用配置里的模型名。
    parser.add_argument("--embed-model", default=None, help="嵌入模型名。")


# 定义 _config_from_args：把命令行参数合并进配置，未指定的用默认值补齐。
def _config_from_args(args: argparse.Namespace) -> RagConfig:
    # 先拿一份默认配置：像先拿一张已经填好默认值的表格。
    defaults = RagConfig()
    # 合并成新配置：命令行给了就用命令行的值，没给就用默认值；dim 用 is not None 判断，因为 0 也会被显式传入。
    return RagConfig(
        store_path=args.store or defaults.store_path,
        embedder=args.embedder or defaults.embedder,
        dim=args.dim if args.dim is not None else defaults.dim,
        base_url=args.base_url or defaults.base_url,
        embed_model=args.embed_model or defaults.embed_model,
    )


# 只有直接运行 python -m rag.cli 时才执行下面代码：被 import 时不会自动跑。
if __name__ == "__main__":
    # 调用 main() 并把返回码交给系统：0 成功、1 运行错误、2 用法错误。
    raise SystemExit(main())
