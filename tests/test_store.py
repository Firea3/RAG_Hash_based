"""store 模块测试：add 校验、余弦 top-k 与排序稳定性、JSON 存取、异常与 clear。

对应用例：TC-STORE-001 ~ TC-STORE-013（docs/04_test_plan.md 第 5.4 节）。
"""

import json
import math
import unittest

from rag.models import Chunk
from rag.store import VectorStore
from tests._support import temp_dir


def _chunk(chunk_id: str, text: str = "片段", source: str = "data/s.md") -> Chunk:
    """构造测试用 Chunk。"""
    return Chunk(chunk_id=chunk_id, doc_id="doc", text=text, source=source)


def _chunk_payload(chunk_id: str) -> dict:
    """构造索引文件中的 chunk 字典。"""
    return {"chunk_id": chunk_id, "doc_id": "doc", "text": "片段", "source": "data/s.md", "metadata": {"chunk_index": 0}}


class VectorStoreAddTest(unittest.TestCase):
    """VectorStore.add 的契约测试。"""

    def setUp(self) -> None:
        self.store = VectorStore(dim=3)

    def test_add_and_len(self) -> None:
        """TC-STORE-001：add 后 len 正确且可检索到全部片段。"""
        self.store.add([_chunk("c1"), _chunk("c2")], [[1, 0, 0], [0, 1, 0]])
        self.assertEqual(len(self.store), 2)
        self.assertEqual(len(self.store.search([1, 0, 0], 10)), 2)

    def test_count_mismatch_raises(self) -> None:
        """TC-STORE-002：chunks 与 vectors 数量不一致抛 ValueError，且不写入任何数据。"""
        with self.assertRaises(ValueError):
            self.store.add([_chunk("c1"), _chunk("c2")], [[1, 0, 0]])
        self.assertEqual(len(self.store), 0)

    def test_dimension_mismatch_raises(self) -> None:
        """TC-STORE-003：向量维度不等于 self.dim 抛 ValueError。"""
        with self.assertRaises(ValueError):
            self.store.add([_chunk("c1")], [[1, 0, 0, 0]])
        with self.assertRaises(ValueError):
            self.store.add([_chunk("c1")], [[1, 0]])


class VectorStoreSearchTest(unittest.TestCase):
    """VectorStore.search 的契约测试。"""

    def setUp(self) -> None:
        self.store = VectorStore(dim=3)
        chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3"), _chunk("c4")]
        vectors = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
        self.store.add(chunks, vectors)

    def test_cosine_order_and_tie_break(self) -> None:
        """TC-STORE-004：按 (-score, chunk_id) 排序，同分时 chunk_id 升序，分数与手算一致。"""
        hits = self.store.search([1, 0, 0], 4)
        self.assertEqual([hit.chunk_id for hit in hits], ["c1", "c2", "c4", "c3"])
        self.assertAlmostEqual(hits[0].score, 1.0, places=6)
        self.assertAlmostEqual(hits[1].score, 1.0, places=6)
        self.assertAlmostEqual(hits[2].score, math.sqrt(2) / 2, places=6)
        self.assertAlmostEqual(hits[3].score, 0.0, places=6)

    def test_top_k_non_positive_returns_empty(self) -> None:
        """TC-STORE-005：top_k <= 0 返回 []。"""
        self.assertEqual(self.store.search([1, 0, 0], 0), [])
        self.assertEqual(self.store.search([1, 0, 0], -1), [])

    def test_empty_store_returns_empty(self) -> None:
        """TC-STORE-006：空库返回 []。"""
        self.assertEqual(VectorStore(dim=3).search([1, 0, 0], 4), [])

    def test_zero_query_vector_returns_empty(self) -> None:
        """TC-STORE-007：全零查询向量返回 []（分母为 0 全部跳过）。"""
        self.assertEqual(self.store.search([0, 0, 0], 4), [])

    def test_query_dimension_mismatch_raises(self) -> None:
        """TC-STORE-008：查询向量维度不符抛 ValueError。"""
        with self.assertRaises(ValueError):
            self.store.search([1, 0, 0, 0], 4)

    def test_top_k_limits_result_count(self) -> None:
        """TC-STORE-004 补充：top_k 限制返回条数且保持排序。"""
        hits = self.store.search([1, 0, 0], 2)
        self.assertEqual([hit.chunk_id for hit in hits], ["c1", "c2"])


class VectorStorePersistTest(unittest.TestCase):
    """VectorStore.save / load / clear 的契约测试。"""

    def setUp(self) -> None:
        self.root = temp_dir(self)
        self.store = VectorStore(dim=3)
        self.store.add([_chunk("c1", text="中文片段")], [[1.0, 0.0, 0.0]])

    def test_save_structure_and_format(self) -> None:
        """TC-STORE-009：自动建父目录；JSON 结构、版本、UTF-8、ensure_ascii=False、indent=2 正确。"""
        path = self.root / "deep" / "nested" / "index.json"
        self.store.save(path)
        self.assertTrue(path.exists())
        raw = path.read_text(encoding="utf-8")
        self.assertIn("中文片段", raw)
        self.assertNotIn("\\u4e2d", raw)
        self.assertIn('"version": 1', raw)
        self.assertIn('\n  "chunks"', raw)
        payload = json.loads(raw)
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload["embedder"], {"name": "hashing", "dim": 3})
        self.assertEqual(len(payload["chunks"]), 1)
        self.assertEqual(payload["chunks"][0]["chunk_id"], "c1")
        self.assertEqual(payload["vectors"], [[1.0, 0.0, 0.0]])

    def test_load_roundtrip(self) -> None:
        """TC-STORE-010：save 后 load 能恢复 chunks、维度、嵌入器名，检索结果一致。"""
        path = self.root / "index.json"
        self.store.save(path)
        restored = VectorStore.load(path)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.embedder_name, "hashing")
        self.assertEqual(restored.dim, 3)
        self.assertEqual(
            [hit.chunk_id for hit in restored.search([1, 0, 0], 4)],
            [hit.chunk_id for hit in self.store.search([1, 0, 0], 4)],
        )

    def test_load_missing_file_raises(self) -> None:
        """TC-STORE-011：文件不存在抛 FileNotFoundError。"""
        with self.assertRaises(FileNotFoundError):
            VectorStore.load(self.root / "nope.json")

    def test_load_invalid_files_raise_value_error(self) -> None:
        """TC-STORE-012：版本非 1、缺字段、数量不一致、维度不一致均抛 ValueError。"""
        cases = {
            "version2": {"version": 2, "embedder": {"name": "hashing", "dim": 3}, "chunks": [], "vectors": []},
            "missing_vectors": {"version": 1, "embedder": {"name": "hashing", "dim": 3}, "chunks": []},
            "count_mismatch": {
                "version": 1,
                "embedder": {"name": "hashing", "dim": 3},
                "chunks": [_chunk_payload("c1")],
                "vectors": [],
            },
            "dim_mismatch": {
                "version": 1,
                "embedder": {"name": "hashing", "dim": 3},
                "chunks": [_chunk_payload("c1")],
                "vectors": [[1.0, 0.0]],
            },
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                path = self.root / f"bad_{name}.json"
                path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                with self.assertRaises(ValueError):
                    VectorStore.load(path)

    def test_clear_keeps_disk_file(self) -> None:
        """TC-STORE-013：clear 清空内存数据但不删除磁盘文件。"""
        path = self.root / "index.json"
        self.store.save(path)
        self.store.clear()
        self.assertEqual(len(self.store), 0)
        self.assertEqual(self.store.search([1, 0, 0], 4), [])
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
