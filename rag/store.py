"""轻量向量存储：JSON 持久化 + 余弦相似度 top-k。"""

# 需要 json 读写索引文件：把片段和向量存成“键: 值”的文本清单。
import json
# 需要 math.sqrt 算向量长度：余弦相似度要除以两个向量的长度。
import math
# 需要 Sequence 类型注解：表示按顺序取元素的一组值，比如列表。
from collections.abc import Sequence
# 需要 asdict 把 dataclass 转成字典：保存 Chunk 时一行就能变成可写 JSON 的结构。
from dataclasses import asdict
# 需要 Path 处理索引文件路径：创建父目录、读写文件都更方便。
from pathlib import Path

# 需要 Chunk、Hit 两个模型：库里存 Chunk，检索返回 Hit。
from rag.models import Chunk, Hit

# 索引文件格式版本号：以后格式变了可以靠它判断，像文件第一页写着“第 1 版”。
INDEX_VERSION = 1


# 定义 VectorStore：内存里的向量库，负责追加、检索和 JSON 存取。
class VectorStore:
    """内存向量库，支持 JSON 存取与确定性排序检索。"""

    # 初始化：记录嵌入器名字和维度，并准备空列表装片段和向量。
    def __init__(self, embedder_name: str = "hashing", dim: int = 256) -> None:
        """初始化。参数：embedder_name 嵌入器名、dim 向量维度；返回：无。"""
        # 维度必须是正整数：0 或负数都没法比较向量。
        if dim <= 0:
            # 报错并告诉当前值：像尺子刻度不对，要立刻修。
            raise ValueError(f"dim 必须为正整数，当前为 {dim}")
        # 记住嵌入器名字：保存索引时写进去，读取时用来确认是同一种嵌入。
        self.embedder_name = embedder_name
        # 记住向量维度：例如 256，后面每个向量都必须是这个长度。
        self.dim = dim
        # 片段列表，一开始为空：每加入一批片段就往这里放。
        self.chunks: list[Chunk] = []
        # 向量列表，和 chunks 一一对应：第 i 个向量描述第 i 个片段。
        self.vectors: list[list[float]] = []

    # 定义 __len__：让 len(store) 直接返回片段数量。
    def __len__(self) -> int:
        """片段数量。参数：无；返回：条数。"""
        # 返回片段列表长度：例如入过 10 块，len(store) 就是 10。
        return len(self.chunks)

    # 定义 add：把一批片段和对应向量追加进库。
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        """追加片段与向量。参数：chunks 片段序列、vectors 向量序列；返回：无。"""
        # 把片段序列转成列表存一份：后面要数数量、做校验。
        chunk_list = list(chunks)
        # 把每个向量转成 float 列表：JSON 读回来的整数也统一成小数，避免类型混乱。
        vector_list = [[float(value) for value in vector] for vector in vectors]
        # 片段数量必须等于向量数量：每块文字都要配一个向量，缺一个都不行。
        if len(chunk_list) != len(vector_list):
            # 数量不一致就报错并列出两个数字：像清单写 10 件却只拿来 8 件。
            raise ValueError(f"片段数量（{len(chunk_list)}）与向量数量（{len(vector_list)}）不一致")
        # 逐个检查向量长度是否等于库的维度。
        for vector in vector_list:
            # 维度不一致就没法比较，直接报错。
            if len(vector) != self.dim:
                # 提示用户删除旧索引重建：像换了新锁芯，就得重新配钥匙。
                raise ValueError(f"向量维度不一致：期望 {self.dim}，实际 {len(vector)}（请删除索引后重建）")
        # 校验全部通过，把片段批量追加到片段列表：extend 像把一摞纸直接倒进抽屉。
        self.chunks.extend(chunk_list)
        # 把向量批量追加到向量列表：位置和上面的片段一一对应。
        self.vectors.extend(vector_list)

    # 定义 search：用查询向量找最相似的 top_k 个片段。
    def search(self, query_vector: Sequence[float], top_k: int = 4) -> list[Hit]:
        """余弦相似度检索。参数：query_vector 查询向量、top_k 返回条数；返回：按相关度排序的 Hit 列表。"""
        # 要的条数不是正数，或者库里没有片段，都直接返回空列表。
        if top_k <= 0 or not self.chunks:
            # 返回空列表：像店里没货，就不必排队结账。
            return []
        # 把查询向量转成 float 列表：统一类型，后面计算更稳。
        query = [float(value) for value in query_vector]
        # 查询向量维度必须和库的维度一致，否则没法逐位比较。
        if len(query) != self.dim:
            # 维度不符就报错并给出两个维度值：像插头规格不对，插不进去。
            raise ValueError(f"查询向量维度不一致：期望 {self.dim}，实际 {len(query)}")
        # 如果查询向量所有分量都是 0（any 为 False），就没有方向可比。
        if not any(query):
            # 返回空列表：像问“空白问题”，没法找相似内容。
            return []
        # 准备空列表装打分后的命中结果。
        scored: list[Hit] = []
        # 把每个片段和它的向量成对取出，逐个算相似度。
        for chunk, vector in zip(self.chunks, self.vectors):
            # 计算查询向量与该向量夹角的余弦值：越大越像，返回 None 表示没法比。
            score = _cosine(query, vector)
            # 能算出分数才收进结果：None 表示有一方是全零向量。
            if score is not None:
                # 把“片段 + 分数”打包成 Hit 放进结果列表。
                scored.append(Hit(chunk=chunk, score=score))
        # 排序规则：先按分数从大到小；分数相同时按片段编号从小到大，保证结果稳定。
        scored.sort(key=lambda hit: (-hit.score, hit.chunk.chunk_id))
        # 只取前 top_k 条返回：例如 top_k=2 就只要最像的两条。
        return scored[:top_k]

    # 定义 save：把当前库保存成 JSON 文件。
    def save(self, path: str | Path) -> None:
        """保存到 JSON 文件。参数：path 索引路径；返回：无。"""
        # 把传入路径转成 Path 对象，方便创建目录和写文件。
        target = Path(path)
        # 先创建父目录：parents=True 连父级一起建，exist_ok=True 表示目录已存在也不报错。
        target.parent.mkdir(parents=True, exist_ok=True)
        # 组装要写入的字典：version 版本号 + embedder 嵌入器信息 + chunks 片段 + vectors 向量，四样缺一不可。
        payload = {
            "version": INDEX_VERSION,
            "embedder": {"name": self.embedder_name, "dim": self.dim},
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "vectors": self.vectors,
        }
        # 转成 JSON 文本（中文原样、缩进 2 格）后用 UTF-8 写入文件：人打开也能看懂，像工整的清单。
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # @classmethod 让这个方法属于类本身：不用先创建对象就能调用 VectorStore.load(...)。
    @classmethod
    # 定义 load：从 JSON 文件恢复出一个 VectorStore。
    def load(cls, path: str | Path) -> "VectorStore":
        """从 JSON 文件恢复向量库。参数：path 索引路径；返回：VectorStore 实例。"""
        # 把传入路径转成 Path 对象。
        target = Path(path)
        # 文件不存在就没法加载。
        if not target.exists():
            # 报 FileNotFoundError 并提示先执行 ingest：像打开空抽屉前，先提醒你去装东西。
            raise FileNotFoundError(f"索引文件不存在：{target}（请先执行 ingest 构建索引）")
        # 尝试读取文件并解析 JSON：文件坏了会抛 JSONDecodeError。
        try:
            # 用 UTF-8 读取文本并解析成 Python 字典。
            payload = json.loads(target.read_text(encoding="utf-8"))
        # 接住 JSON 语法错误，换成带路径的中文 ValueError。
        except json.JSONDecodeError as exc:
            # 报错并带上原始原因：像老师说“第 3 行少了个逗号”。
            raise ValueError(f"索引文件不是合法 JSON：{target}（{exc}）") from exc
        # 检查顶层是不是字典、版本号是不是 1：版本不同说明格式对不上。
        if not isinstance(payload, dict) or payload.get("version") != INDEX_VERSION:
            # 报错并说明要求的版本号：像确认收纳盒标签是不是同一版。
            raise ValueError(f"索引版本或结构非法：{target}（要求 version={INDEX_VERSION}）")
        # 取出嵌入器信息字段：缺失就是 None。
        info = payload.get("embedder")
        # 取出片段列表字段：缺失就是 None。
        raw_chunks = payload.get("chunks")
        # 取出向量列表字段：缺失就是 None。
        raw_vectors = payload.get("vectors")
        # 三个字段类型都必须正确：info 是字典，两个列表都必须是列表。
        if not isinstance(info, dict) or not isinstance(raw_chunks, list) or not isinstance(raw_vectors, list):
            # 有字段缺失或类型不对就报错：像清单缺页，不能继续清点。
            raise ValueError(f"索引缺少 embedder / chunks / vectors 字段：{target}")
        # 从嵌入器信息里取名字。
        name = info.get("name")
        # 从嵌入器信息里取维度。
        dim = info.get("dim")
        # 校验：名字必须是字符串，维度必须是正整数；bool 是 int 的子类，要单独排除 True/False。
        if not isinstance(name, str) or not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            # 信息非法就报错，避免用错误的维度去比较向量。
            raise ValueError(f"索引中的 embedder 信息非法：{target}")
        # 片段数量必须等于向量数量：每块文字配一个向量。
        if len(raw_chunks) != len(raw_vectors):
            # 数量不一致就报错并列出两个数：像清单和实物对不上。
            raise ValueError(f"索引中片段数量（{len(raw_chunks)}）与向量数量（{len(raw_vectors)}）不一致：{target}")
        # 用读到的名字和维度创建一个空库：先造好架子，再把东西摆上去。
        store = cls(name, dim)
        # 把 JSON 里的片段逐条还原成 Chunk 对象。
        store.chunks = [_chunk_from_json(item, target) for item in raw_chunks]
        # 把 JSON 里的向量逐条校验并转成 float 列表。
        store.vectors = [_vector_from_json(item, dim, target) for item in raw_vectors]
        # 返回恢复好的向量库：调用者可以继续检索或追加。
        return store

    # 定义 clear：清空内存里的片段和向量。
    def clear(self) -> None:
        """清空内存中的片段与向量（不删除磁盘文件）。参数：无；返回：无。"""
        # 清空片段列表：像把抽屉里的卡片全倒出来。
        self.chunks.clear()
        # 清空向量列表：和片段保持同步清空。
        self.vectors.clear()


# 定义 _cosine：计算两个向量的余弦相似度，长度为 0 时返回 None。
def _cosine(left: list[float], right: list[float]) -> float | None:
    # 算左向量长度：每个分量平方求和再开根号，例如 (3,4) 长度是 5。
    left_norm = math.sqrt(sum(value * value for value in left))
    # 算右向量长度：方法同上。
    right_norm = math.sqrt(sum(value * value for value in right))
    # 任一长度为 0 都不能做除法。
    if left_norm == 0.0 or right_norm == 0.0:
        # 返回 None 表示“没法比”：像一张白纸和任何画都谈不上像不像。
        return None
    # 点积除以两个长度：结果在 -1 到 1 之间，越接近 1 越像；例如 (1,0) 与 (0.8,0.6) 得 0.8。
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


# 定义 _chunk_from_json：把一个 JSON 字典还原成 Chunk 对象。
def _chunk_from_json(item: object, target: Path) -> Chunk:
    # 元素必须是字典，否则结构不对。
    if not isinstance(item, dict):
        # 报错并带上索引文件路径：像发现一页格式不对，告诉你出自哪本书。
        raise ValueError(f"索引 chunks 元素结构非法：{target}")
    # 尝试取字段并组装 Chunk：缺字段或类型不对会在下面统一接住。
    try:
        # metadata 缺省用空字典：老索引可能没有这个字段。
        metadata = item.get("metadata") or {}
        # metadata 必须是字典：列表或字符串都不合法。
        if not isinstance(metadata, dict):
            # 主动抛 TypeError，让下面统一转成更容易读懂的 ValueError。
            raise TypeError("metadata 必须是字典")
        # 组装并返回 Chunk：四个字符串字段缺一不可，用 str() 统一转成文本。
        return Chunk(
            chunk_id=str(item["chunk_id"]),
            doc_id=str(item["doc_id"]),
            text=str(item["text"]),
            source=str(item["source"]),
            metadata=dict(metadata),
        )
    # 接住缺字段（KeyError）、类型不对（TypeError）、转字符串出错（ValueError）。
    except (KeyError, TypeError, ValueError) as exc:
        # 换成带路径的 ValueError：提示用户索引文件损坏，需要重建。
        raise ValueError(f"索引 chunks 元素缺字段或字段类型非法：{target}") from exc


# 定义 _vector_from_json：把一个 JSON 列表还原成指定维度的向量。
def _vector_from_json(item: object, dim: int, target: Path) -> list[float]:
    # 元素必须是列表，且长度必须等于维度。
    if not isinstance(item, list) or len(item) != dim:
        # 维度不符就报错：像拼图块数和图纸对不上。
        raise ValueError(f"索引向量维度与 embedder.dim={dim} 不一致：{target}")
    # 尝试把每个元素都转成 float：有非数字元素会抛异常。
    try:
        # 返回转好的向量：例如 [0.1, -0.2, ...]。
        return [float(value) for value in item]
    # 接住类型错误和数值转换错误。
    except (TypeError, ValueError) as exc:
        # 换成带路径的 ValueError：提示索引文件被改坏了。
        raise ValueError(f"索引向量包含非数值元素：{target}") from exc
