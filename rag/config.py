"""RAG 运行配置。"""

# 从标准库 dataclasses 取 dataclass：它能自动生成初始化方法，少写很多重复代码。
from dataclasses import dataclass


# dataclass(frozen=True) 自动生成构造函数并禁止修改字段：像填好并盖章的通知单，之后不能再改字。
@dataclass(frozen=True)
# 定义配置类 RagConfig：一次 RAG 运行需要的全部参数都装在这里，像一张“设置清单”。
class RagConfig:
    """一次 RAG 运行的配置。

    参数：
        chunk_size: 片段最大字符数。
        chunk_overlap: 相邻片段重叠字符数。
        top_k: 默认检索返回条数。
        store_path: 索引 JSON 文件路径。
        embedder: 嵌入方式，hashing 或 openai。
        dim: 向量维度。
        base_url: OpenAI 兼容接口地址。
        embed_model: 嵌入模型名。
        api_key: 接口密钥，为空时读取环境变量。
    """

    # 每个片段最多 420 个字符：像把长文章装进 420 格的小盒子，装不下就换下一个盒子。
    chunk_size: int = 420
    # 相邻片段重叠 80 个字符：像两页书交界处重复 80 字，防止一句话被切成两半后接不上。
    chunk_overlap: int = 80
    # 默认一次返回 4 条检索结果：像问同学借笔记时先拿最像的 4 页来看。
    top_k: int = 4
    # 索引默认存到 data/index.json：所有片段和向量都记在这个文件里，像一本总账本。
    store_path: str = "data/index.json"
    # 默认使用 hashing（离线哈希）嵌入：不需要网络和密钥，像自带铅笔，随时能写字。
    embedder: str = "hashing"
    # 默认向量维度为 256：每段文字用 256 个数字来描述，像用 256 个刻度来量一段文字。
    dim: int = 256
    # 默认的 OpenAI 兼容接口地址：只有选 openai 嵌入时才会用到。
    base_url: str = "https://api.deepseek.com"
    # 默认嵌入模型名：调接口时告诉对方“用哪个模型来算向量”。
    embed_model: str = "deepseek-embedding"
    # 接口密钥默认 None（先不填）：真正调用时再去环境变量里找，像先不写密码，用的时候再查密码本。
    api_key: str | None = None
