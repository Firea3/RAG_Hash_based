"""tool 模块测试：TOOL_SPEC 结构、run_tool 参数校验与返回结构、tool_schema.json 语义相等。

对应用例：TC-TOOL-001 ~ TC-TOOL-009（docs/04_test_plan.md 第 5.6 节）。
"""

import json
import unittest
from pathlib import Path
from unittest import mock

from rag.config import RagConfig
from rag.pipeline import Rag
from rag.tool import TOOL_SPEC, run_tool
from tests._support import PROJECT_ROOT, temp_dir

SCHEMA_PATH = PROJECT_ROOT / "skills" / "deepseek-rag" / "tool_schema.json"


class ToolSpecTest(unittest.TestCase):
    """TOOL_SPEC 的契约测试。"""

    def test_tool_spec_structure(self) -> None:
        """TC-TOOL-001：function calling 结构、函数名、参数约束完全符合契约。"""
        self.assertEqual(TOOL_SPEC["type"], "function")
        function = TOOL_SPEC["function"]
        self.assertEqual(function["name"], "search_knowledge_base")
        self.assertIsInstance(function["description"], str)
        self.assertTrue(function["description"])
        parameters = function["parameters"]
        self.assertEqual(parameters["type"], "object")
        self.assertEqual(parameters["required"], ["query"])
        query = parameters["properties"]["query"]
        self.assertEqual(query["type"], "string")
        top_k = parameters["properties"]["top_k"]
        self.assertEqual(top_k["type"], "integer")
        self.assertEqual(top_k["minimum"], 1)
        self.assertEqual(top_k["maximum"], 10)
        self.assertEqual(top_k["default"], 4)

    def test_tool_schema_json_matches_spec(self) -> None:
        """TC-TOOL-009：skills/deepseek-rag/tool_schema.json 与 TOOL_SPEC JSON 语义相等。"""
        if not SCHEMA_PATH.exists():
            self.skipTest(f"tool_schema.json 尚未产出：{SCHEMA_PATH}")
        loaded = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(loaded, TOOL_SPEC)


class RunToolTest(unittest.TestCase):
    """run_tool 的契约测试。"""

    def setUp(self) -> None:
        self.root = temp_dir(self)
        self.index_path = self.root / "index.json"
        kb = self.root / "kb"
        kb.mkdir()
        (kb / "a.md").write_text("知识库检索需要先切片再计算向量。" * 4, encoding="utf-8")
        (kb / "b.md").write_text("引用规则：每个结论后用编号标注来源。" * 4, encoding="utf-8")
        self.rag = Rag(RagConfig(store_path=str(self.index_path), chunk_size=50, chunk_overlap=10, top_k=4))
        self.rag.ingest([kb])

    def test_run_tool_with_dict_arguments(self) -> None:
        """TC-TOOL-002：dict 参数返回 {"query","results"}，results 每项含四个键。"""
        result = run_tool({"query": "知识库检索", "top_k": 2}, rag=self.rag)
        self.assertEqual(result["query"], "知识库检索")
        self.assertIsInstance(result["results"], list)
        self.assertLessEqual(len(result["results"]), 2)
        self.assertTrue(result["results"])
        for item in result["results"]:
            self.assertEqual(set(item), {"text", "source", "score", "chunk_id"})

    def test_run_tool_with_json_string(self) -> None:
        """TC-TOOL-003：JSON 字符串参数被解析，返回结构与 dict 调用一致。"""
        as_dict = run_tool({"query": "引用规则", "top_k": 1}, rag=self.rag)
        as_json = run_tool(json.dumps({"query": "引用规则", "top_k": 1}), rag=self.rag)
        self.assertEqual(as_json["query"], as_dict["query"])
        self.assertEqual(set(as_json), {"query", "results"})
        self.assertEqual(len(as_json["results"]), len(as_dict["results"]))

    def test_invalid_json_raises_value_error(self) -> None:
        """TC-TOOL-004：非法 JSON 字符串抛 ValueError；JSON 标量非法。"""
        with self.assertRaises(ValueError):
            run_tool("{not json}")
        with self.assertRaises(ValueError):
            run_tool("123")
        with self.assertRaises(ValueError):
            run_tool("[]")

    def test_invalid_query_raises_value_error(self) -> None:
        """TC-TOOL-005：query 缺失/非字符串/空串/纯空白均抛 ValueError。"""
        for arguments in ({}, {"query": None}, {"query": 123}, {"query": ""}, {"query": "   "}):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    run_tool(arguments, rag=self.rag)

    def test_invalid_top_k_raises_value_error(self) -> None:
        """TC-TOOL-006：top_k 必须是 1~10 的整数；bool、浮点、字符串、越界均拒绝。"""
        for top_k in (0, 11, -1, 4.0, "4", True):
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    run_tool({"query": "知识库检索", "top_k": top_k}, rag=self.rag)

    def test_result_shape_and_score_precision(self) -> None:
        """TC-TOOL-007：score 保留 6 位小数；无命中时 results 为 []。"""
        result = run_tool({"query": "知识库检索"}, rag=self.rag)
        self.assertEqual(set(result), {"query", "results"})
        for item in result["results"]:
            self.assertIsInstance(item["score"], float)
            self.assertAlmostEqual(item["score"], round(item["score"], 6), places=12)

        empty_rag = Rag(RagConfig(store_path=str(self.root / "empty.json")))
        empty_result = run_tool({"query": "任何问题"}, rag=empty_rag)
        self.assertEqual(empty_result, {"query": "任何问题", "results": []})

    def test_default_rag_created_when_none(self) -> None:
        """TC-TOOL-008：未传 rag 时新建默认 Rag()，返回结构合法。"""
        with mock.patch("rag.tool.Rag") as rag_class:
            rag_class.return_value.query.return_value = []
            result = run_tool({"query": "问题"})
        rag_class.assert_called_once_with()
        self.assertEqual(result, {"query": "问题", "results": []})


if __name__ == "__main__":
    unittest.main()
