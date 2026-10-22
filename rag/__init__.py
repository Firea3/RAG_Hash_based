"""deepseek-agent-rag：为 DeepSeek Agent 提供本地知识库检索（RAG）能力。"""

# 导入配置类 RagConfig：用它可以修改 chunk_size、top_k 等参数，像点菜时把默认口味改成自己的偏好。
from rag.config import RagConfig
# 导入三个数据模型：Document 是原文、Chunk 是切出的片段、Hit 是检索命中，像三张用途不同的表格。
from rag.models import Chunk, Document, Hit
# 导入门面类 Rag：加载、分块、嵌入、存储、检索全由它串起来，像一条完整的作业流水线。
from rag.pipeline import Rag
# 导入工具协议和执行函数：TOOL_SPEC 告诉 Agent 怎么调用，run_tool 负责真正执行检索。
from rag.tool import TOOL_SPEC, run_tool

# 版本号按“主版本.次版本.修订号”写：0.1.0 表示第一个小版本，像作业本第一册第一课。
__version__ = "0.1.0"
# 声明这个包对外公开的名字：from rag import * 时只带出这些，像店门口挂着“本店主营”的牌子。
__all__ = ["Rag", "RagConfig", "Document", "Chunk", "Hit", "TOOL_SPEC", "run_tool", "__version__"]
