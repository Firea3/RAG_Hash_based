"""中文友好的文本分块：段落 -> 句子 -> 贪心打包。"""

# 需要 re 正则模块：用“句末标点”作为切分点，把长段落切成一句一句。
import re

# 需要 Chunk、Document 两个数据模型：输入文档、输出片段。
from rag.models import Chunk, Document

# 编译一个正则：在句末标点“后面”切一刀；(?<=…) 是“回头看一眼”的零宽断言，切点本身不丢字。
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])")


# 定义 split_text：把长文本切成不超过 chunk_size、相邻有重叠的片段列表。
def split_text(text: str, chunk_size: int = 420, chunk_overlap: int = 80) -> list[str]:
    """把文本切成不超过 chunk_size 的重叠片段。参数：text 原文、chunk_size 块大小、chunk_overlap 重叠；返回：片段列表。"""
    # 先检查参数是否合法：块大小必须为正，重叠不能为负且必须小于块大小。
    _validate(chunk_size, chunk_overlap)
    # 统一换行符并去掉首尾空白：Windows 的 \r\n 和旧 Mac 的 \r 都变成 \n，像把不同格式的作业统一成一种纸。
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # 如果归一化后是空文本，就没有内容可切。
    if not normalized:
        # 返回空列表表示“没有片段”：像一张空白纸，不需要装订成册。
        return []
    # 准备空列表装“句子单元”：后面会把句子贪心打包成片段。
    units: list[str] = []
    # 先按空行切段落：空行通常表示分段，像把文章按自然段分成一摞纸。
    for paragraph in normalized.split("\n\n"):
        # 再按句末标点把每个段落切成句子：正则在标点后切，所以句子会带着标点。
        for sentence in _SENTENCE_END.split(paragraph.strip()):
            # 去掉句子两端的空白：像把每张纸条四周的毛边裁掉。
            sentence = sentence.strip()
            # 只保留非空句子：空字符串不产生单元。
            if sentence:
                # 超长句子先硬切成若干块，再加入单元列表；extend 像把一摞纸直接倒进箱子。
                units.extend(_hard_split(sentence, chunk_size))
    # 用贪心打包把句子单元拼成最终片段：尽量塞满一块，同时保留尾部重叠。
    return _pack(units, chunk_size, chunk_overlap)


# 定义 chunk_document：把一篇 Document 切成带编号的 Chunk 列表。
def chunk_document(doc: Document, chunk_size: int = 420, chunk_overlap: int = 80) -> list[Chunk]:
    """把文档切成带编号的片段。参数：doc 文档、chunk_size 块大小、chunk_overlap 重叠；返回：Chunk 列表。"""
    # 先把文档正文切成若干文本片段：复用上面的 split_text。
    pieces = split_text(doc.text, chunk_size, chunk_overlap)
    # 把每个文本片段包装成 Chunk：编号 = 文档编号 + "#" + 序号，元数据合并原标签并追加 chunk_index。
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{index}",
            doc_id=doc.doc_id,
            text=piece,
            source=doc.source,
            metadata={**doc.metadata, "chunk_index": index},
        )
        for index, piece in enumerate(pieces)
    ]


# 定义 _validate：检查分块参数是否合法，不合法就报 ValueError 并说明怎么改。
def _validate(chunk_size: int, chunk_overlap: int) -> None:
    # 块大小必须为正整数：0 或负数都没法装文字。
    if chunk_size <= 0:
        # 报错并给出当前值：像登记表不合格时，告诉你是哪一项写错了。
        raise ValueError(f"chunk_size 必须为正整数，当前为 {chunk_size}")
    # 重叠必须满足 0 <= 重叠 < 块大小：等于或超过块大小会导致切片永远前进不了。
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        # 报错并说明合法范围：像说明书告诉你重叠最多只能比块小 1。
        raise ValueError(f"chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size，当前为 {chunk_overlap}")


# 定义 _hard_split：把超长句子按固定长度硬切成若干块。
def _hard_split(segment: str, chunk_size: int) -> list[str]:
    # 从 0 开始每隔 chunk_size 取一段：例如长度 900、块 420 的句子会切成 420/420/60 三段。
    return [segment[start : start + chunk_size] for start in range(0, len(segment), chunk_size)]


# 定义 _pack：贪心打包算法，把句子单元拼成尽量满且带尾部重叠的片段。
def _pack(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    # 准备空列表装最终片段：像先摆好一排空袋子。
    chunks: list[str] = []
    # current 是“正在装的袋子”，一开始是空的。
    current = ""
    # 逐个处理句子单元：能塞就塞，塞不下就封袋并开新袋。
    for unit in units:
        # 袋子还空着时，直接把当前单元放进去。
        if not current:
            # 第一个单元直接成为袋子的内容：像新袋子里先放第一块饼干。
            current = unit
        # 如果“现有内容 + 1 个换行 + 新句子”超过块大小，说明这个袋子装不下了。
        elif len(current) + 1 + len(unit) > chunk_size:
            # 先把当前袋子封口，放进结果列表。
            chunks.append(current)
            # 取当前内容的最后 chunk_overlap 个字符作为重叠尾巴；重叠为 0 时尾巴是空字符串。
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            # 新袋子的候选内容 = 重叠尾巴 + 换行 + 新句子；没有尾巴时就只有新句子。
            candidate = f"{tail}\n{unit}" if tail else unit
            # 如果候选内容没超过块大小就用它，否则放弃重叠只放新句子，保证每块都不超长。
            current = candidate if len(candidate) <= chunk_size else unit
        # 其他情况说明袋子还装得下：把新句子用换行接到后面。
        else:
            # 用换行连接新旧句子：像往袋子里再放一块饼干，中间垫一张纸分隔。
            current = f"{current}\n{unit}"
    # 循环结束后，如果还有没封口的袋子，也要收进结果。
    if current:
        # 把最后一个袋子封口：像把最后一页纸也装进相册。
        chunks.append(current)
    # 返回全部片段：每个长度都 <= chunk_size，相邻片段之间带重叠。
    return chunks
