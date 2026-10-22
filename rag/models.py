"""领域数据模型：Document / Chunk / Hit。"""

# 从标准库 dataclasses 取 dataclass 和 field：dataclass 自动生成构造方法，field 用来给默认值加规则。
from dataclasses import dataclass, field


# 让 Document 自动拥有构造函数，并且创建后不能修改：像打印好的档案卡，填完就封存。
@dataclass(frozen=True)
# 定义 Document（文档）：一篇原始文档 = 编号 + 正文 + 来源 + 附加信息，像一本书的借阅卡。
class Document:
    """一篇原始文档。参数：doc_id 文档编号、text 原文、source 来源路径、metadata 附加信息。"""

    # 文档编号，例如 "data/knowledge_base/agent_faq.md"：像每本书唯一的索书号。
    doc_id: str
    # 文档正文，例如 "DeepSeek Agent RAG 使用手册……"：这是真正要被切块和检索的文字。
    text: str
    # 来源路径，例如 "data/knowledge_base/agent_faq.md"：回答时用它告诉用户这句话出自哪个文件。
    source: str
    # 附加信息字典，默认空字典：可以放自定义标签，像档案卡背面的备注栏。
    metadata: dict = field(default_factory=dict)


# 让 Chunk 自动生成构造函数并禁止修改：像装好一小段文字的密封袋，贴上标签后不再拆改。
@dataclass(frozen=True)
# 定义 Chunk（片段）：长文档切出来的一个小段，是检索的最小单位，像把长文章剪成一张张小卡片。
class Chunk:
    """文档切出的一个片段。参数：chunk_id 片段编号、doc_id 文档编号、text 片段正文、source 来源路径、metadata 附加信息。"""

    # 片段编号，例如 "data/knowledge_base/agent_faq.md#0"：文档编号加“#序号”，像同一本书的第 0 页。
    chunk_id: str
    # 这个片段来自哪篇文档：例如 "data/knowledge_base/agent_faq.md"。
    doc_id: str
    # 片段正文，例如 "如何接入 DeepSeek Agent？……"：检索命中后返回给 Agent 的就是它。
    text: str
    # 片段来源路径：和 doc_id 一样指向原文件，方便回答时标注引用。
    source: str
    # 附加信息，默认空字典：入库时会写入 {"chunk_index": 0}，表示“这是第 0 块”。
    metadata: dict = field(default_factory=dict)


# 让 Hit 自动生成构造函数并禁止修改：一次命中就是“片段 + 分数”的固定组合。
@dataclass(frozen=True)
# 定义 Hit（命中）：检索结果里的一条，像考试排名表上的一行“姓名 + 分数”。
class Hit:
    """一次检索命中。参数：chunk 命中的片段、score 余弦相似度。"""

    # 命中的片段对象：里面包含正文、来源和编号。
    chunk: Chunk
    # 相似度分数，越接近 1 越相关：例如 0.812345 表示“挺像”。
    score: float

    # @property 把方法伪装成属性：调用时写 hit.text，不用写 hit.text()，像直接查看表格的一列。
    @property
    # 定义 text 属性：直接取片段正文，少写一层 hit.chunk.text。
    def text(self) -> str:
        """片段正文。参数：无；返回：chunk.text。"""
        # 把内层片段的 text 原样转出去：像传话时把原话一字不改地告诉下一个人。
        return self.chunk.text

    # 同样是属性：调用者写 hit.source 就能拿到来源路径。
    @property
    # 定义 source 属性：转发片段的来源路径。
    def source(self) -> str:
        """来源路径。参数：无；返回：chunk.source。"""
        # 把内层片段的 source 转出去：例如 "data/knowledge_base/agent_faq.md"。
        return self.chunk.source

    # 同样是属性：调用者写 hit.chunk_id 就能拿到片段编号。
    @property
    # 定义 chunk_id 属性：转发片段的编号。
    def chunk_id(self) -> str:
        """片段编号。参数：无；返回：chunk.chunk_id。"""
        # 把内层片段的 chunk_id 转出去：例如 "data/knowledge_base/agent_faq.md#0"。
        return self.chunk.chunk_id
