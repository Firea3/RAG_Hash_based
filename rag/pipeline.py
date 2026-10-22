"""RAG 管道门面：ingest / query / context / save。"""

# 需要 Sequence 类型注解：表示按顺序取元素的一组值，比如路径列表。
from collections.abc import Sequence
# 需要 Path 判断索引文件是否存在：存在就加载，不存在就新建。
from pathlib import Path

# 需要 chunk_document 把文档切成片段。
from rag.chunker import chunk_document
# 需要 RagConfig 配置类：没传配置时用默认配置。
from rag.config import RagConfig
# 需要 Embedder 类型和 build_embedder 工厂：按配置创建嵌入器。
from rag.embedder import Embedder, build_embedder
# 需要 load_documents 把路径加载成文档。
from rag.loaders import load_documents
# 需要 Hit 类型：query 返回的检索结果。
from rag.models import Hit
# 需要 VectorStore 向量库：负责存储和检索。
from rag.store import VectorStore


# 定义 Rag 门面类：把加载、分块、嵌入、存储、检索串成一条完整流水线。
class Rag:
    """RAG 门面：串联加载、分块、嵌入、存储与检索。"""

    # 初始化流水线：准备好配置、嵌入器和向量库三样东西。
    def __init__(
        self,
        config: RagConfig | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        """初始化管道。参数：config 配置、embedder 嵌入器、store 向量库；返回：无。"""
        # 没传配置就用默认配置：例如 chunk_size=420、top_k=4。
        self.config = config or RagConfig()
        # 没传嵌入器就按配置创建：例如配置默认的 hashing 嵌入器。
        self.embedder = embedder or build_embedder(self.config)
        # 如果调用者直接给了向量库，就直接用它。
        if store is not None:
            # 记下调用者给的向量库：方便测试注入或复用已有索引。
            self.store = store
        # 否则看看配置里的索引文件是否存在：存在就加载。
        elif Path(self.config.store_path).exists():
            # 从 JSON 文件恢复向量库：像按收纳清单把玩具摆回架子。
            self.store = VectorStore.load(self.config.store_path)
        # 索引文件不存在，就新建一个空库。
        else:
            # 用嵌入器的名字和维度创建空库：例如 hashing、256 维。
            self.store = VectorStore(self.embedder.name, self.embedder.dim)
        # 最后检查维度是否一致：库的维度必须等于嵌入器维度，否则向量对不上。
        if self.store.dim != self.embedder.dim:
            # 维度不一致就报错，并给出可操作的修复方法：删掉旧索引后重新 ingest。
            raise ValueError(
                f"索引维度（{self.store.dim}）与嵌入维度（{self.embedder.dim}）不一致："
                f"请删除 {self.config.store_path} 后重新 ingest"
            )

    # 定义 ingest：把文档灌入索引并保存到磁盘。
    def ingest(self, paths: Sequence[str | Path], rebuild: bool = False) -> dict[str, int]:
        """把文档灌入索引并保存。参数：paths 文件或目录序列、rebuild 是否先清空；返回：文档数与片段数。"""
        # 需要全量重建时，先清空内存里的旧片段和向量。
        if rebuild:
            # 清空旧数据：像重新整理书架前，先把旧书全拿下来。
            self.store.clear()
        # 加载路径下的全部文档：Markdown/TXT/JSON 都会变成 Document。
        documents = load_documents(paths)
        # 用双重列表推导把每篇文档切成片段再拍平：外层走文档，内层走该文档的每个片段。
        chunks = [
            chunk for document in documents
            for chunk in chunk_document(document, self.config.chunk_size, self.config.chunk_overlap)
        ]
        # 批量把片段正文喂给嵌入器，得到一一对应的向量列表。
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        # 把片段和向量追加进向量库：数量和维度都会在这里校验。
        self.store.add(chunks, vectors)
        # 把更新后的索引保存成 JSON 文件：下次启动可以直接加载。
        self.store.save(self.config.store_path)
        # 返回统计信息：例如 {"documents": 3, "chunks": 12}。
        return {"documents": len(documents), "chunks": len(chunks)}

    # 定义 query：把问题变成向量，再在库里找最相似的片段。
    def query(self, question: str, top_k: int | None = None) -> list[Hit]:
        """检索与问题最相关的片段。参数：question 问题、top_k 返回条数；返回：Hit 列表。"""
        # 问题为空或只有空白时，直接返回空列表，不必浪费一次嵌入计算。
        if not question or not question.strip():
            # 返回空列表：像没写问题就交卷，当然没有答案。
            return []
        # 把问题嵌成向量：embed 返回列表，取第一个就是问题的向量。
        query_vector = self.embedder.embed([question])[0]
        # 在库里检索：top_k 传了就用它，没传就用配置里的默认值（4）。
        return self.store.search(query_vector, top_k or self.config.top_k)

    # 定义 context：把检索结果拼成带引用编号的上下文文本。
    def context(self, question: str, top_k: int | None = None) -> str:
        """生成带引用编号的上下文。参数：question 问题、top_k 返回条数；返回：上下文文本（无命中为空串）。"""
        # 先检索出命中的片段列表。
        hits = self.query(question, top_k)
        # 一条都没命中，就没有上下文可拼。
        if not hits:
            # 返回空字符串：像查不到资料，就老实说没有。
            return ""
        # 每条命中拼成一块：第一行是引用编号和来源，第二行是片段正文。
        blocks = [f"[{index}] 来源：{hit.source}\n{hit.text}" for index, hit in enumerate(hits, start=1)]
        # 用两个换行连接各块，形成段落分明的上下文：像把几张资料卡隔开摆好。
        return "\n\n".join(blocks)

    # 定义 save：把当前索引保存到配置指定的路径。
    def save(self) -> None:
        """保存索引到配置中的路径。参数：无；返回：无。"""
        # 交给向量库保存：路径来自配置，例如 data/index.json。
        self.store.save(self.config.store_path)
