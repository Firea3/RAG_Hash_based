"""隔离与离线守卫测试：真实 data/index.json 不被触碰；hashing 链路不打开 socket。

对应用例：TC-GUARD-001 ~ TC-GUARD-002（docs/04_test_plan.md 第 5.12 节）。
"""

import unittest
from unittest import mock

from rag.config import RagConfig
from rag.pipeline import Rag
from rag.tool import run_tool
from tests._support import PROJECT_ROOT, fingerprint, temp_dir


class GuardTest(unittest.TestCase):
    """安全红线测试。"""

    def setUp(self) -> None:
        self.root = temp_dir(self)
        self.kb = self.root / "kb"
        self.kb.mkdir()
        (self.kb / "a.md").write_text("知识库检索与引用规则说明。" * 5, encoding="utf-8")

    def _make_rag(self) -> Rag:
        """构造指向临时索引的 RAG 管道。"""
        config = RagConfig(store_path=str(self.root / "index.json"), chunk_size=50, chunk_overlap=10, top_k=2)
        rag = Rag(config)
        rag.ingest([self.kb])
        return rag

    def test_real_index_untouched(self) -> None:
        """TC-GUARD-001：完整跑一遍 ingest/query/tool 后，真实 data/index.json 内容指纹不变。"""
        real_index = PROJECT_ROOT / "data" / "index.json"
        before = fingerprint(real_index)
        rag = self._make_rag()
        self.assertTrue(rag.query("知识库检索"))
        run_tool({"query": "知识库检索"}, rag=rag)
        self.assertEqual(fingerprint(real_index), before, "测试改动了真实 data/index.json")

    def test_no_socket_connection_needed(self) -> None:
        """TC-GUARD-002：禁用 socket 后 hashing 全链路仍可用（证明完全离线，AC5）。"""
        with mock.patch("socket.socket", side_effect=AssertionError("测试禁止联网")):
            rag = self._make_rag()
            hits = rag.query("知识库检索")
            result = run_tool({"query": "知识库检索", "top_k": 1}, rag=rag)
            context = rag.context("引用规则")
        self.assertTrue(hits)
        self.assertTrue(result["results"])
        self.assertIn("来源：", context)


if __name__ == "__main__":
    unittest.main()
