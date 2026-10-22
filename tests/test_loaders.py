"""loaders 模块测试：目录递归、后缀过滤、编码容错、JSON 三种元素形态。

对应用例：TC-LOAD-001 ~ TC-LOAD-012（docs/04_test_plan.md 第 5.1 节）。
"""

import json
import unittest
from pathlib import Path

from rag.loaders import SUPPORTED_SUFFIXES, load_documents
from tests._support import temp_dir


class LoadersTest(unittest.TestCase):
    """rag.loaders.load_documents 的契约测试。"""

    def setUp(self) -> None:
        self.root = temp_dir(self)

    def _write(self, relative: str, content: str, encoding: str = "utf-8") -> Path:
        """在临时目录下写一个文本文件并返回路径。"""
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return path

    def test_loads_markdown_file(self) -> None:
        """TC-LOAD-001：单个 .md 文件加载为 1 个 Document，source 用 posix 形式。"""
        path = self._write("a.md", "# 标题\n\n中文内容。\n")
        docs = load_documents([path])
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc.text, "# 标题\n\n中文内容。\n")
        self.assertEqual(doc.doc_id, doc.source)
        self.assertEqual(doc.source, path.as_posix())
        self.assertIn("/", doc.source)
        self.assertNotIn("\\", doc.source)
        self.assertEqual(dict(doc.metadata), {})

    def test_loads_txt_file(self) -> None:
        """TC-LOAD-002：.txt 文件被支持。"""
        path = self._write("b.txt", "纯文本内容")
        docs = load_documents([path])
        self.assertEqual([doc.text for doc in docs], ["纯文本内容"])
        self.assertTrue(docs[0].source.endswith(".txt"))

    def test_loads_markdown_suffix_file(self) -> None:
        """TC-LOAD-003：.markdown 文件被支持。"""
        path = self._write("c.markdown", "markdown 后缀")
        docs = load_documents([path])
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0].source.endswith(".markdown"))

    def test_directory_recursive_sorted_and_ignored(self) -> None:
        """TC-LOAD-004：目录递归加载、按路径排序、忽略隐藏文件与特定目录。"""
        self._write("a.md", "A")
        self._write("b.txt", "B")
        self._write("sub/c.md", "C")
        self._write(".hidden.md", "隐藏文件")
        self._write("__pycache__/p.md", "缓存")
        self._write(".git/g.md", "git")
        self._write("_acl_recovery/r.md", "ACL 临时目录")
        docs = load_documents([self.root])
        sources = [doc.source for doc in docs]
        self.assertEqual(len(sources), 3)
        self.assertEqual(sources, sorted(sources))
        self.assertTrue(sources[0].endswith("/a.md"))
        self.assertTrue(sources[1].endswith("/b.txt"))
        self.assertTrue(sources[2].endswith("/sub/c.md"))
        for source in sources:
            self.assertNotIn(".hidden", source)
            self.assertNotIn("__pycache__", source)
            self.assertNotIn(".git", source)
            self.assertNotIn("_acl_recovery", source)

    def test_unsupported_suffix_skipped(self) -> None:
        """TC-LOAD-005：不支持的后缀静默跳过，不抛异常。"""
        self._write("a.md", "有效")
        self._write("b.pdf", "PDF 占位")
        self._write("c.bin", "二进制")
        self._write("d.png", "图片")
        docs = load_documents([self.root])
        self.assertEqual(len(docs), 1)
        self.assertTrue(docs[0].source.endswith("a.md"))

    def test_missing_path_raises_file_not_found(self) -> None:
        """TC-LOAD-006：路径不存在抛 FileNotFoundError，且不产生任何文件。"""
        missing = self.root / "nope.md"
        with self.assertRaises(FileNotFoundError):
            load_documents([missing])
        self.assertFalse(missing.exists())

    def test_empty_file_skipped(self) -> None:
        """TC-LOAD-007：空文件（含纯空白）被跳过。"""
        self._write("empty.md", "")
        self._write("blank.txt", "   \n\t  ")
        self._write("ok.md", "有内容")
        docs = load_documents([self.root])
        self.assertEqual([doc.text for doc in docs], ["有内容"])

    def test_invalid_utf8_bytes_are_replaced(self) -> None:
        """TC-LOAD-008：非 UTF-8 字节用 errors="replace" 读取，不抛编码异常。"""
        path = self.root / "bad.txt"
        path.write_bytes(b"\xff\xfe\x00abc")
        docs = load_documents([path])
        self.assertEqual(len(docs), 1)
        self.assertIsInstance(docs[0].text, str)
        self.assertIn("\ufffd", docs[0].text)

    def test_json_list_of_strings(self) -> None:
        """TC-LOAD-009：JSON 顶层字符串列表逐项成 Document，doc_id 带下标。"""
        path = self.root / "list.json"
        path.write_text(json.dumps(["第一段", "第二段"], ensure_ascii=False), encoding="utf-8")
        docs = load_documents([path])
        self.assertEqual([doc.text for doc in docs], ["第一段", "第二段"])
        self.assertEqual(docs[0].doc_id, f"{path.as_posix()}#0")
        self.assertEqual(docs[1].doc_id, f"{path.as_posix()}#1")
        self.assertEqual(docs[0].metadata["json_index"], 0)
        self.assertEqual(docs[1].metadata["json_index"], 1)
        self.assertEqual(docs[0].source, path.as_posix())

    def test_json_items_dict_priority_and_skip(self) -> None:
        """TC-LOAD-010：{"items": [...]} 字典优先级 text > content > question/answer，非法项跳过，json_index 用原下标。"""
        path = self.root / "mixed.json"
        items = [
            {"text": "A"},
            {"content": "B"},
            {"question": "Q", "answer": "An"},
            {"other": "X"},
            123,
            {"content": "B2", "metadata": {"k": "v"}},
        ]
        path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
        docs = load_documents([path])
        self.assertEqual([doc.text for doc in docs], ["A", "B", "问：Q\n答：An", "B2"])
        self.assertEqual([doc.metadata["json_index"] for doc in docs], [0, 1, 2, 5])
        self.assertEqual(docs[3].doc_id, f"{path.as_posix()}#5")
        self.assertEqual(docs[3].metadata, {"k": "v", "json_index": 5})

    def test_invalid_json_raises_value_error_with_path(self) -> None:
        """TC-LOAD-011：非法 JSON 抛 ValueError，消息包含文件路径；结构不支持的 JSON 同样报错。"""
        path = self.root / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError) as ctx:
            load_documents([path])
        self.assertIn("broken.json", str(ctx.exception))

        unsupported = self.root / "unsupported.json"
        unsupported.write_text(json.dumps({"foo": 1}), encoding="utf-8")
        with self.assertRaises(ValueError) as ctx2:
            load_documents([unsupported])
        self.assertIn("unsupported.json", str(ctx2.exception))

    def test_supported_suffixes_exact(self) -> None:
        """TC-LOAD-012：SUPPORTED_SUFFIXES 精确等于契约规定的四个后缀。"""
        self.assertEqual(SUPPORTED_SUFFIXES, {".md", ".markdown", ".txt", ".json"})


if __name__ == "__main__":
    unittest.main()
