"""pipeline 模块测试：构造与维度一致性、ingest/query/context/save、rebuild、空输入。

对应用例：TC-RAG-001 ~ TC-RAG-010（docs/04_test_plan.md 第 5.5 节）。
所有索引路径都指向临时目录，绝不读写真实 data/index.json。
"""

import json
import unittest
from pathlib import Path

from rag.config import RagConfig
from rag.models import Hit
from rag.pipeline import Rag
from rag.store import VectorStore
from tests._support import temp_dir


class _CountingEmbedder:
    """记录 embed 调用次数的测试替身（离线、确定性）。"""

    name = "counting"
    dim = 4

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts):
        """统计调用次数并返回全零向量。"""
        self.calls += 1
        return [[0.0] * self.dim for _ in texts]


class RagTest(unittest.TestCase):
    """rag.pipeline.Rag 的契约测试。"""

    def setUp(self) -> None:
        self.root = temp_dir(self)
        self.index_path = self.root / "index.json"
        self.kb = self.root / "kb"
        self.kb.mkdir()
        # 两篇文档，内容足够长，保证能切出多个片段。
        for name in ("a.md", "b.md"):
            text = "知识库检索需要先把文档切片，再计算向量。" * 6
            (self.kb / name).write_text(text, encoding="utf-8")

    def _config(self, **overrides) -> RagConfig:
        """构造指向临时索引的配置。"""
        defaults = {"store_path": str(self.index_path), "chunk_size": 50, "chunk_overlap": 10, "top_k": 2}
        defaults.update(overrides)
        return RagConfig(**defaults)

    def test_init_without_index_creates_empty_store(self) -> None:
        """TC-RAG-001：索引不存在时新建空 store，构造不落盘，query 返回 []。"""
        rag = Rag(self._config())
        self.assertEqual(rag.query("问题"), [])
        self.assertEqual(rag.store.dim, rag.embedder.dim)
        self.assertEqual(rag.store.embedder_name, "hashing")
        self.assertFalse(self.index_path.exists())

    def test_init_dimension_mismatch_raises(self) -> None:
        """TC-RAG-002：索引维度与嵌入维度不一致抛 ValueError，并提示删除索引后重建。"""
        VectorStore(dim=8).save(self.index_path)
        with self.assertRaises(ValueError) as ctx:
            Rag(self._config(dim=256))
        self.assertIn("维度", str(ctx.exception))

    def test_ingest_returns_counts_and_saves(self) -> None:
        """TC-RAG-003：ingest 返回 documents/chunks 计数并保存索引。"""
        rag = Rag(self._config())
        summary = rag.ingest([self.kb])
        self.assertEqual(summary["documents"], 2)
        self.assertGreaterEqual(summary["chunks"], 2)
        self.assertIsInstance(summary["documents"], int)
        self.assertIsInstance(summary["chunks"], int)
        self.assertTrue(self.index_path.exists())
        self.assertEqual(len(rag.store), summary["chunks"])
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["chunks"]), summary["chunks"])

    def test_ingest_rebuild_does_not_duplicate(self) -> None:
        """TC-RAG-004：rebuild=True 先清空，重复入库不会累积重复数据。"""
        rag = Rag(self._config())
        first = rag.ingest([self.kb])
        second = rag.ingest([self.kb], rebuild=True)
        self.assertEqual(first["chunks"], second["chunks"])
        self.assertEqual(len(rag.store), second["chunks"])

    def test_ingest_empty_directory_saves_empty_index(self) -> None:
        """TC-RAG-005：空数据正常返回 0/0，索引仍被保存。"""
        empty = self.root / "empty_kb"
        empty.mkdir()
        rag = Rag(self._config())
        summary = rag.ingest([empty])
        self.assertEqual(summary, {"documents": 0, "chunks": 0})
        self.assertTrue(self.index_path.exists())
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["chunks"], [])
        self.assertEqual(payload["vectors"], [])

    def test_query_blank_returns_empty_without_embedding(self) -> None:
        """TC-RAG-006：空问题/纯空白返回 []，且不调用嵌入。"""
        embedder = _CountingEmbedder()
        rag = Rag(self._config(), embedder=embedder)
        self.assertEqual(rag.query(""), [])
        self.assertEqual(rag.query("   \n\t "), [])
        self.assertEqual(embedder.calls, 0)

    def test_query_respects_top_k(self) -> None:
        """TC-RAG-007：返回 Hit 列表，条数受 config.top_k 与显式 top_k 限制。"""
        rag = Rag(self._config(top_k=2))
        rag.ingest([self.kb])
        default_hits = rag.query("知识库检索")
        self.assertTrue(default_hits)
        self.assertLessEqual(len(default_hits), 2)
        for hit in default_hits:
            self.assertIsInstance(hit, Hit)
            self.assertIsInstance(hit.text, str)
            self.assertIsInstance(hit.source, str)
            self.assertIsInstance(hit.score, float)
            self.assertIsInstance(hit.chunk_id, str)
        self.assertLessEqual(len(rag.query("知识库检索", top_k=1)), 1)

    def test_context_format_and_empty(self) -> None:
        """TC-RAG-008：context 首行 [n] 来源：source，序号从 1 递增、块间空行；无命中返回空串。"""
        rag = Rag(self._config())
        rag.ingest([self.kb])
        context = rag.context("知识库检索")
        self.assertTrue(context)
        blocks = context.split("\n\n")
        for index, block in enumerate(blocks, start=1):
            lines = block.split("\n", 1)
            self.assertTrue(lines[0].startswith(f"[{index}] 来源："), block)
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[1].strip())

        empty_rag = Rag(self._config(store_path=str(self.root / "empty.json")))
        self.assertEqual(empty_rag.context("任何问题"), "")
        self.assertEqual(empty_rag.context(""), "")

    def test_save_and_reload_persists_index(self) -> None:
        """TC-RAG-009：save 后新实例能从磁盘恢复并检索到内容。"""
        rag = Rag(self._config())
        rag.ingest([self.kb])
        rag.save()
        reloaded = Rag(self._config())
        self.assertEqual(len(reloaded.store), len(rag.store))
        self.assertTrue(reloaded.query("知识库检索"))

    def test_ingest_missing_path_propagates_file_not_found(self) -> None:
        """TC-RAG-010：知识库路径不存在时透传 FileNotFoundError。"""
        rag = Rag(self._config())
        with self.assertRaises(FileNotFoundError):
            rag.ingest([self.root / "nope"])


if __name__ == "__main__":
    unittest.main()
