"""chunker 模块测试：参数校验、换行归一化、贪心打包、重叠上限、硬切、编号规则。

对应用例：TC-CHUNK-001 ~ TC-CHUNK-010（docs/04_test_plan.md 第 5.2 节）。
"""

import unittest

from rag.chunker import chunk_document, split_text
from rag.models import Document


def _overlap_length(left: str, right: str) -> int:
    """计算 left 的后缀与 right 的前缀的最长重叠长度。"""
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


class SplitTextTest(unittest.TestCase):
    """rag.chunker.split_text 的契约测试。"""

    @staticmethod
    def _long_text(sentences: int = 60) -> str:
        """构造包含多句中文的长文本。"""
        return "".join(f"第{i}段内容，讲的是知识库检索与引用规则。" for i in range(sentences))

    def test_long_text_within_size_limit(self) -> None:
        """TC-CHUNK-001：长文本分块非空，且每块满足 0 < len <= chunk_size。"""
        text = self._long_text()
        chunks = split_text(text, 420, 80)
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertGreater(len(chunk), 0)
            self.assertLessEqual(len(chunk), 420)
        joined = "".join(chunks)
        for keyword in ("知识库检索", "引用规则"):
            self.assertIn(keyword, joined)

    def test_invalid_parameters_raise_value_error(self) -> None:
        """TC-CHUNK-002：chunk_size<=0、chunk_overlap<0、overlap>=size 均抛 ValueError。"""
        cases = [(0, 0), (100, -1), (100, 100), (100, 150)]
        for chunk_size, chunk_overlap in cases:
            with self.subTest(chunk_size=chunk_size, chunk_overlap=chunk_overlap):
                with self.assertRaises(ValueError):
                    split_text("abc", chunk_size, chunk_overlap)

    def test_normalizes_newlines_and_trims(self) -> None:
        """TC-CHUNK-003：\\r\\n 与 \\r 统一为 \\n，去首尾空白。"""
        chunks = split_text("  第一段\r\n第二段\r第三段  ", 420, 80)
        joined = "".join(chunks)
        self.assertNotIn("\r", joined)
        self.assertEqual(joined.replace("\n", ""), "第一段第二段第三段")
        self.assertEqual(joined, joined.strip())

    def test_empty_or_whitespace_returns_empty_list(self) -> None:
        """TC-CHUNK-004：空文本、纯空白文本返回 []。"""
        self.assertEqual(split_text("", 420, 80), [])
        self.assertEqual(split_text("   \n\t ", 420, 80), [])

    def test_adjacent_overlap_within_limit(self) -> None:
        """TC-CHUNK-005：相邻片段重叠长度不超过 chunk_overlap，且块长不超过 chunk_size。"""
        chunks = split_text(self._long_text(), 100, 20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)
        for left, right in zip(chunks, chunks[1:]):
            self.assertLessEqual(_overlap_length(left, right), 20)

    def test_hard_split_long_segment(self) -> None:
        """TC-CHUNK-006：无标点超长片段被硬切，每块不超过 chunk_size。"""
        chunks = split_text("a" * 500, 100, 20)
        self.assertEqual(len(chunks), 5)
        for chunk in chunks:
            self.assertEqual(len(chunk), 100)

    def test_short_paragraphs_are_merged(self) -> None:
        """TC-CHUNK-008：短段落贪心打包为 1 块，不丢内容、无空块。"""
        chunks = split_text("第一段。\n\n第二段。", 420, 80)
        self.assertEqual(len(chunks), 1)
        self.assertIn("第一段", chunks[0])
        self.assertIn("第二段", chunks[0])

    def test_deterministic_output(self) -> None:
        """TC-CHUNK-009：同样输入连续调用结果完全一致（无随机性）。"""
        text = self._long_text()
        self.assertEqual(split_text(text, 100, 20), split_text(text, 100, 20))

    def test_overlap_near_size_boundary_terminates(self) -> None:
        """TC-CHUNK-010：chunk_overlap = chunk_size - 1 合法，正常终止且块长受限。"""
        chunks = split_text(self._long_text(), 100, 99)
        self.assertTrue(chunks)
        self.assertLess(len(chunks), 200)
        for chunk in chunks:
            self.assertGreater(len(chunk), 0)
            self.assertLessEqual(len(chunk), 100)


class ChunkDocumentTest(unittest.TestCase):
    """rag.chunker.chunk_document 的契约测试。"""

    def test_chunk_ids_and_metadata(self) -> None:
        """TC-CHUNK-007：chunk_id 为 doc_id#i，metadata 带 chunk_index 并保留原 metadata。"""
        doc = Document(doc_id="d1", text="第一句话。第二句话。第三句话。第四句话。", source="d1.md", metadata={"k": "v"})
        chunks = chunk_document(doc, 12, 5)
        self.assertGreater(len(chunks), 1)
        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, f"d1#{index}")
            self.assertEqual(chunk.doc_id, "d1")
            self.assertEqual(chunk.source, "d1.md")
            self.assertEqual(chunk.metadata["k"], "v")
            self.assertEqual(chunk.metadata["chunk_index"], index)
            self.assertLessEqual(len(chunk.text), 12)


if __name__ == "__main__":
    unittest.main()
