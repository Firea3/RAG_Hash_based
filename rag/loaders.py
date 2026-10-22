"""文档加载：把 Markdown / TXT / JSON 读成 Document。"""

# 需要 json 模块把 JSON 文本解析成字典、列表：像把压缩饼干拆开，看清里面的配料。
import json
# 需要 os.walk 递归遍历目录：它会自动进入子文件夹，像一层层打开抽屉找文件。
import os
# 需要 Sequence 类型注解：表示“可以按顺序取元素的一组值”，比如列表或元组。
from collections.abc import Sequence
# 需要 Path 处理路径：拼接、取后缀、读写文件都比字符串方便，像给地址簿配了一台导航仪。
from pathlib import Path

# 需要 Document 数据模型：加载的最终结果就是一篇篇 Document。
from rag.models import Document

# 支持的文档后缀集合：只有 .md/.markdown/.txt/.json 会被加载，其他文件像不合规的快递会被退回。
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".json"}
# 遍历目录时要跳过的文件夹：缓存、版本库、恢复目录里的东西不是知识库内容，像打扫时跳过杂物间。
_IGNORED_DIRS = {"__pycache__", ".git", "_acl_recovery"}


# 定义 load_documents：把一串文件或目录路径变成 Document 列表，是加载阶段的总入口。
def load_documents(paths: Sequence[str | Path]) -> list[Document]:
    """加载路径下的全部文档。参数：paths 文件或目录序列；返回：Document 列表。"""
    # 先准备一个空列表装文档：像先拿一个大纸箱，读到一篇就往里放一篇。
    documents: list[Document] = []
    # 逐个处理传入的路径：字符串和 Path 对象都可以，像收货时一条条核对快递单。
    for raw in paths:
        # 把输入统一转成 Path 对象：例如 "data/kb" 变成可以调方法的路径对象。
        path = Path(raw)
        # 路径不存在就直接报错：像按地址找不到门牌号，必须停下来问清楚，不能假装已经送达。
        if not path.exists():
            # 抛出 FileNotFoundError 并说明是哪个路径：像找不到门牌号时，把地址念给用户听，方便他核对。
            raise FileNotFoundError(f"路径不存在：{path}（请检查路径是否正确）")
        # 是目录就递归列出里面的文件，是单个文件就直接用它：像寄一箱书和寄一本书的区别。
        files = _iter_files(path) if path.is_dir() else [path]
        # 逐个读取文件：每个文件可能产出 0 篇、1 篇或多篇文档（一个 JSON 文件可以包含多条）。
        for file_path in files:
            # 把该文件加载出的文档一次性追加进大纸箱：extend 像把一摞纸直接倒进箱子，不用一张张拆。
            documents.extend(_load_file(file_path))
    # 返回全部文档：调用者拿去切块，例如 [Document(...), Document(...)]。
    return documents


# 定义 _iter_files：递归找出目录下所有受支持的文档文件，并排好序返回。
def _iter_files(root: Path) -> list[Path]:
    # 准备空列表装文件路径：像先拿一个文件筐，找到一个就放一个。
    files: list[Path] = []
    # os.walk 会一层层遍历目录，每次给出“当前目录、子目录名列表、文件名列表”三样东西。
    for dir_path, dir_names, file_names in os.walk(root):
        # 原地筛选子目录：用切片赋值改掉整个列表，walk 之后就只进入这些“可见”子目录。
        dir_names[:] = sorted(name for name in dir_names if _is_visible_dir(name))
        # 逐个检查当前目录下的文件：看名字和后缀是否符合要求。
        for name in file_names:
            # 取小写后缀：例如 "FAQ.MD" 会变成 ".md"，大小写不同也算同一种格式。
            suffix = Path(name).suffix.lower()
            # 只收“不以点开头”且后缀受支持的文件：隐藏文件和 .exe 之类的都跳过，像只收语文作业。
            if not name.startswith(".") and suffix in SUPPORTED_SUFFIXES:
                # 把“当前目录 + 文件名”拼成完整路径，放进文件筐。
                files.append(Path(dir_path) / name)
    # 按路径排序后返回：保证每次加载顺序一致，像按学号排队交作业，结果不会忽上忽下。
    return sorted(files, key=lambda item: item.as_posix())


# 定义 _is_visible_dir：判断一个子目录要不要继续往里找。
def _is_visible_dir(name: str) -> bool:
    # 不以点开头（不是隐藏目录）且不在黑名单里：两个条件同时满足才放行，像安检两道关都过才让进。
    return not name.startswith(".") and name not in _IGNORED_DIRS


# 定义 _load_file：读取单个文件，按后缀决定是当纯文本还是 JSON 解析。
def _load_file(path: Path) -> list[Document]:
    # 后缀不在支持列表里就直接返回空列表：不支持的文件静默跳过，不报错。
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        # 返回空列表表示“这个文件没有产生文档”：像不认识的快递直接放回架子上。
        return []
    # 用 UTF-8 读取文件，坏字节用“�”替换：不抛编码错误，保证程序继续跑，像错字用方框代替。
    text = path.read_text(encoding="utf-8", errors="replace")
    # 如果去掉空白后什么都没有，就是空文件，同样跳过。
    if not text.strip():
        # 返回空列表表示“空文件不收”：像一张白纸，不值得归档。
        return []
    # 把路径转成斜杠形式作为来源：例如 data\knowledge_base\a.md 变成 data/knowledge_base/a.md，跨平台都一致。
    source = path.as_posix()
    # 后缀是 .json 时交给专门的 JSON 解析函数：一个 JSON 文件可能包含很多条问答。
    if path.suffix.lower() == ".json":
        # 返回 JSON 解析出的文档列表：可能有多条，也可能一条都没有。
        return _load_json(path, text)
    # 普通文本（Markdown/TXT）整个文件就是一篇文档：doc_id 和 source 都用斜杠路径。
    return [Document(doc_id=source, text=text, source=source)]


# 定义 _load_json：解析 JSON 文件，把里面的条目逐条变成 Document。
def _load_json(path: Path, text: str) -> list[Document]:
    # 先取斜杠路径作为来源：后面会拼进每条文档的编号，例如 "faq.json#0"。
    source = path.as_posix()
    # 尝试把 JSON 文本解析成 Python 对象：成功就是列表或字典，失败会抛 JSONDecodeError。
    try:
        # payload 可能是列表（一排条目）或字典（带 "items" 隔层的礼盒）。
        payload = json.loads(text)
    # 接住 JSON 语法错误，换成带文件名和行号的中文 ValueError：方便用户定位哪里写错了。
    except json.JSONDecodeError as exc:
        # 报错信息包含文件路径、出错行号和具体原因，像老师批作业写“第 3 行少逗号”。
        raise ValueError(f"JSON 文件解析失败：{path}（第 {exc.lineno} 行：{exc.msg}）") from exc
    # 统一取出条目列表：顶层是字典就取 "items"，否则把顶层本身当列表；像先打开礼盒的隔层。
    items = payload.get("items") if isinstance(payload, dict) else payload
    # 取出来的东西必须是列表，否则结构不支持，直接报错说明正确格式。
    if not isinstance(items, list):
        # 报错并提示两种合法写法：顶层列表，或 {"items": [...]}，像说明书告诉你只能按两种方式装。
        raise ValueError(f'JSON 结构不支持：{source}（顶层需为列表或 {{"items": [...]}}）')
    # 准备空列表装文档：每个条目解析成功就放一篇。
    documents: list[Document] = []
    # 逐条处理并编号：enumerate 给出从 0 开始的序号，像给每块拼图编号 0 号、1 号……
    for index, item in enumerate(items):
        # 从条目里提取正文：字符串直接可用，字典按 text/content/问答规则找，找不到返回 None。
        body = _item_text(item)
        # 正文为 None 说明这条不合法，没有可检索的文字。
        if body is None:
            # 跳过这一条，继续处理下一条：像筛豆子时把石子挑出去。
            continue
        # 条目是字典就复制它的 metadata，否则用空字典：像把原包装上的标签撕下来贴到新袋子上。
        metadata = dict(item.get("metadata") or {}) if isinstance(item, dict) else {}
        # 在 metadata 里记下它在 JSON 里的序号：例如第 2 条记 {"json_index": 2}，方便回溯原文位置。
        metadata["json_index"] = index
        # 生成 Document 并放进列表：doc_id 用“来源#序号”保证每条都不同，像同一本书的第 0 页、第 1 页。
        documents.append(Document(doc_id=f"{source}#{index}", text=body, source=source, metadata=metadata))
    # 返回解析好的全部文档：调用者拿到 [Document(...), ...]。
    return documents


# 定义 _item_text：从 JSON 的一个条目里提取正文文字，提取不到就返回 None。
def _item_text(item: object) -> str | None:
    # 情况一：条目本身就是字符串，直接当正文返回，例如 "RAG 是什么？"。
    if isinstance(item, str):
        # 返回这段字符串：像收到一张已经写好的便签，直接贴进档案。
        return item
    # 情况二：条目不是字典（比如数字、列表），没有可用的文字结构。
    if not isinstance(item, dict):
        # 返回 None 表示“提取不到”：像包装里没有说明书。
        return None
    # 依次尝试 text、content 两个字段：先找首选的，再找备选的。
    for key in ("text", "content"):
        # 取出该字段的值：没有这个字段时得到 None。
        value = item.get(key)
        # 值必须是字符串才算有效正文，数字或列表都不行。
        if isinstance(value, str):
            # 找到有效字符串就立即返回：像找钥匙先看第一个口袋，找到就不用翻第二个。
            return value
    # 再尝试“问答”格式：分别取出 question 和 answer 两个字段。
    question, answer = item.get("question"), item.get("answer")
    # 两个字段都必须是字符串，才能拼成正文。
    if isinstance(question, str) and isinstance(answer, str):
        # 拼成两行“问：…”“答：…”：检索时问句和答案都会被搜到。
        return f"问：{question}\n答：{answer}"
    # 所有规则都不满足，返回 None 表示这条跳过。
    return None
