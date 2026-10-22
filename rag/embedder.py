"""可插拔嵌入：离线确定性 hashing 嵌入与 OpenAI 兼容 /embeddings 接口。"""

# 需要 hashlib 算 blake2b 指纹：同一段文字永远得到同一串字节，像给文字盖上指纹章。
import hashlib
# 需要 json 序列化请求体、解析响应：和网络接口之间用 JSON 这种“键: 值”的清单传递数据。
import json
# 需要 math.log 算 1+ln(tf) 权重、math.sqrt 算向量长度：像开方和乘除要用的计算器。
import math
# 需要 os.environ 读取环境变量里的 API Key：像从密码本上找账号密码。
import os
# 需要 urllib.request 发 HTTP 请求：标准库自带的上网工具，不装第三方包也能调用接口。
import urllib.request
# 需要 Counter 统计每个 2-gram/3-gram 出现次数：像数一数每个字符片段出现了几次。
from collections import Counter
# 需要 Sequence 类型注解：表示有序的一组值，比如文本列表。
from collections.abc import Sequence
# 需要 Protocol 定义“协议”：只规定嵌入器必须具备哪些属性和方法，不管具体怎么实现。
from typing import Protocol

# 需要 RagConfig 配置类：build_embedder 根据它决定用哪种嵌入方式。
from rag.config import RagConfig


# 定义 Embedder 协议：一个嵌入器至少要提供 name、dim 和 embed 三样东西。
class Embedder(Protocol):
    """嵌入器协议。"""

    # 嵌入器的名字，例如 "hashing" 或 "openai"：像工具上贴的型号标签。
    name: str
    # 向量维度，例如 256：表示每段文字用多少个数字来描述。
    dim: int

    # 声明 embed 方法：把一批文本变成一批向量，具体实现由实现类负责。
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把文本列表转成向量列表。参数：texts 文本序列；返回：向量列表。"""
        # 三个点表示“这里先空着，由实现类补上”：像装修图纸上标着“待施工”。
        ...


# 定义 HashingEmbedder：不联网的哈希嵌入器，靠字符片段指纹把文字变成向量。
class HashingEmbedder:
    """离线确定性哈希嵌入（blake2b 特征哈希），跨进程结果一致。"""

    # 给这个嵌入器起固定名字：配置里写 "hashing" 就会选中它。
    name = "hashing"

    # 初始化：确定向量有多少维，维度不合法就直接报错。
    def __init__(self, dim: int = 256) -> None:
        """初始化。参数：dim 向量维度；返回：无。"""
        # 维度必须是正整数：0 维或负数都没地方放特征。
        if dim <= 0:
            # 报错并告诉当前值：像尺子刻度标错了，要立刻纠正。
            raise ValueError(f"dim 必须为正整数，当前为 {dim}")
        # 把维度记在对象上：例如 dim=256，后面每个向量都有 256 个格子。
        self.dim = dim

    # 实现接口方法：把一批文本逐个转成向量。
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把文本列表转成向量列表。参数：texts 文本序列；返回：L2 归一化向量列表。"""
        # 用列表推导式对每段文字调用 _embed_one：像流水线上逐件加工，最后输出一批成品。
        return [self._embed_one(text) for text in texts]

    # 定义 _embed_one：把单段文字变成 dim 维的 L2 归一化向量。
    def _embed_one(self, text: str) -> list[float]:
        # 先创建一个全 0 的向量：dim=256 时就是 256 个 0.0，像摆好 256 个空格子。
        vector = [0.0] * self.dim
        # 统计这段文字的小写 2-gram 和 3-gram 出现次数，再逐个字符片段处理。
        for gram, count in _gram_counts(text.lower()).items():
            # 用 blake2b 给这个字符片段算 8 字节指纹：像给每块拼图盖一枚永不改变的章。
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            # 把 8 字节按“小端”拼成一个 64 位整数：像把 8 块积木从右往左叠成一串数字。
            number = int.from_bytes(digest, "little")
            # 用最高位（第 63 位）决定符号：最高位是 1 就取 -1，否则 +1，像每个字符片段自己抛硬币选方向。
            sign = -1.0 if number >> 63 else 1.0
            # 用整数对维度取余找到格子编号，再按 1+ln(出现次数) 加权累加：例如 300%256=44，就加到第 44 格。
            vector[number % self.dim] += sign * (1.0 + math.log(count))
        # 把向量长度缩放成 1 后返回：以后比较方向时就不用再除一次长度了。
        return _l2_normalize(vector)


# 定义 OpenAICompatibleEmbedder：通过 OpenAI 兼容的 /embeddings 接口做嵌入。
class OpenAICompatibleEmbedder:
    """调用 OpenAI 兼容 /embeddings 接口做嵌入。"""

    # 给这个嵌入器起固定名字：配置里写 "openai" 就会选中它。
    name = "openai"

    # 初始化：保存接口地址、模型名、密钥、维度和超时时间。
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dim: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        """初始化。参数：base_url 接口根地址、model 模型名、api_key 密钥、dim 维度、timeout 超时秒数；返回：无。"""
        # 去掉地址末尾的斜杠，避免拼出 https://x//embeddings 这种双斜杠：像统一门牌号的写法。
        self.base_url = base_url.rstrip("/")
        # 记住模型名：请求时要告诉对方用哪个模型算向量。
        self.model = model
        # 记住传入的密钥：可能是配置给的，也可能是 None（后面再去环境变量里找）。
        self.api_key = api_key
        # 记住期望维度：接口返回的维度必须和它对上，否则报错。
        self.dim = dim
        # 记住超时秒数：网络太慢就中断，避免程序一直卡住像死机一样。
        self.timeout = timeout

    # 实现接口方法：调用接口把一批文本变成向量。
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把文本列表转成向量列表。参数：texts 文本序列；返回：向量列表。"""
        # 空列表直接返回空列表，不必发请求：像没有快递要寄就不用跑邮局。
        if not texts:
            # 返回空列表：调用者拿到 []，不会触发任何网络访问。
            return []
        # 组装请求头：Authorization 带上密钥，Content-Type 说明发的是 JSON。
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        # 组装请求体：model 指定模型，input 是文本列表；list(texts) 先转成真正的列表再发。
        payload = {"model": self.model, "input": list(texts)}
        # 把请求发到“根地址 + /embeddings”，拿到响应字典：这是唯一真正联网的一步。
        data = _post_json(f"{self.base_url}/embeddings", payload, headers, self.timeout)
        # 按约定结构解析响应，并检查条数、维度是否正确，最后返回向量列表。
        return self._parse(data, len(texts))

    # 定义 _api_key：按优先级找密钥，找不到就报错提示用户怎么设置。
    def _api_key(self) -> str:
        # 优先级：传入的 api_key → 环境变量 RAG_API_KEY → 环境变量 OPENAI_API_KEY，取第一个非空值。
        key = self.api_key or os.environ.get("RAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        # 三种来源都没有拿到密钥，就没法调用接口。
        if not key:
            # 报 RuntimeError 并给出解决办法：像门禁提示“请刷卡，或去前台登记”。
            raise RuntimeError("缺少 API Key：请传入 api_key 或设置环境变量 RAG_API_KEY / OPENAI_API_KEY")
        # 把找到的密钥返回给调用方：请求头的 Bearer 后面就填它。
        return key

    # 定义 _parse：解析接口响应，校验条数与维度，出错时换成好懂的中文报错。
    def _parse(self, data: object, expected: int) -> list[list[float]]:
        # 试着从 data["data"] 里逐条取 embedding 并转成 float：结构不对会抛异常，下面统一接住。
        try:
            # 列表推导式逐条取向量：例如 data["data"][0]["embedding"] -> [0.01, -0.02, ...]。
            vectors = [[float(value) for value in item["embedding"]] for item in data["data"]]  # type: ignore[index]
        # 接住四种可能的错误：缺键、越界、类型不对、数值转不了，都说明响应结构非法。
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            # 换成 RuntimeError，并把原始响应带上，方便排查接口到底返回了什么。
            raise RuntimeError(f"嵌入接口响应结构非法：{data!r}") from exc
        # 检查返回条数和每条维度：必须与请求的文本数量、配置的 dim 完全一致。
        if len(vectors) != expected or any(len(vector) != self.dim for vector in vectors):
            # 不一致就报错，防止错误向量写进索引：像收货时数量不对就拒收。
            raise RuntimeError(f"嵌入接口返回条数或维度不符：期望 {expected} 条、每条约 {self.dim} 维")
        # 校验通过，返回解析好的向量列表。
        return vectors


# 定义 _gram_counts：统计文字里所有 2-gram 和 3-gram 的出现次数。
def _gram_counts(text: str) -> Counter[str]:
    # 如果文字不足 2 个字符（0 或 1 个），没法组成 2-gram，就按单字符统计。
    if len(text) < 2:
        # Counter(text) 会数每个字符出现几次：例如 "猫" -> {"猫": 1}。
        return Counter(text)
    # 统计所有相邻两个字符组成的片段：例如 "我爱吃鱼" -> "我爱"、"爱吃"、"吃鱼"。
    counts: Counter[str] = Counter(text[index : index + 2] for index in range(len(text) - 1))
    # 文字长度够 3 时，再补充统计所有相邻三个字符的片段。
    if len(text) >= 3:
        # 例如 "我爱吃鱼" 的 3-gram 是 "我爱吃"、"爱吃鱼"，用 update 累加进同一个计数器。
        counts.update(text[index : index + 3] for index in range(len(text) - 2))
    # 返回计数器：键是字符片段，值是出现次数，加权时用来算 1+ln(tf)。
    return counts


# 定义 _l2_normalize：把向量长度缩放到 1，只保留方向。
def _l2_normalize(vector: list[float]) -> list[float]:
    # 计算长度：每个分量平方后求和再开根号，例如 (3,4) 的长度是 5。
    norm = math.sqrt(sum(value * value for value in vector))
    # 全零向量长度为 0，不能做除法。
    if norm == 0.0:
        # 直接返回原向量（全 0）：像一张白纸没有方向，保持全 0 即可。
        return vector
    # 每个分量都除以长度：例如 (3,4) 变成 (0.6,0.8)，长度正好是 1。
    return [value / norm for value in vector]


# 定义 _post_json：发送 JSON POST 请求并解析 JSON 响应，所有网络访问都集中在这里，便于测试替换。
def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: float) -> dict:
    """发送 JSON POST 请求并解析 JSON 响应。参数：url 地址、payload 请求体、headers 请求头、timeout 超时；返回：响应字典。"""
    # 把请求体转成 JSON 文本再编码成 UTF-8 字节：网络只认字节，像把信装进信封再寄。
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    # 构造请求对象：地址、数据、请求头、方法都打包好，像填好一张快递单。
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    # with 会在用完后自动关闭连接：像借完书自动归还，不会一直占着。
    with urllib.request.urlopen(request, timeout=timeout) as response:
        # 读取响应字节，按 UTF-8 解码，再解析成字典返回：像拆信后读里面的清单。
        return json.loads(response.read().decode("utf-8"))


# 定义 build_embedder：按配置挑一个嵌入器并创建它。
def build_embedder(config: RagConfig) -> Embedder:
    """按配置构造嵌入器。参数：config 配置；返回：Embedder 实例。"""
    # 配置写 "hashing" 就用离线哈希嵌入器：不需要网络，随时可用。
    if config.embedder == "hashing":
        # 创建 HashingEmbedder，维度取配置里的 dim：例如 dim=256。
        return HashingEmbedder(config.dim)
    # 配置写 "openai" 就用接口嵌入器：需要网络和 API Key。
    if config.embedder == "openai":
        # 把地址、模型、密钥、维度交给接口嵌入器：像把外卖地址、口味、电话填进订单。
        return OpenAICompatibleEmbedder(config.base_url, config.embed_model, config.api_key, config.dim)
    # 两种都不是就报错，并告诉用户可选值：像点菜时报“菜单上只有米饭和面条”。
    raise ValueError(f"不支持的嵌入方式：{config.embedder}（可选 hashing 或 openai）")
