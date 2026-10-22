"""端到端测试（AC2）：只读仓库示例知识库，索引写入临时目录。

对应用例：TC-E2E-001 ~ TC-E2E-002（docs/04_test_plan.md 第 5.11 节）。
"""

import contextlib
import io
import json
import os
import re
import unittest

from rag.cli import main
from tests._support import PROJECT_ROOT, fingerprint, temp_dir


class IntegrationTest(unittest.TestCase):
    """示例数据 ingest -> query -> context 全链路。"""

    def setUp(self) -> None:
        self.root = temp_dir(self)
        self.store = self.root / "index.json"
        self.real_index = PROJECT_ROOT / "data" / "index.json"
        # 用例内使用相对路径 data/knowledge_base，需在项目根目录执行。
        self._old_cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        self.addCleanup(os.chdir, self._old_cwd)
        # 守卫：真实索引（若存在）在用例执行前后必须完全一致。
        before = fingerprint(self.real_index)
        self.addCleanup(self._assert_real_index_unchanged, before)

    def _assert_real_index_unchanged(self, before) -> None:
        """断言真实 data/index.json 未被测试读写。"""
        self.assertEqual(fingerprint(self.real_index), before, "测试改动了真实 data/index.json")

    def _call(self, argv):
        """调用 CLI 并捕获输出。"""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def _ingest(self) -> str:
        """用示例知识库构建临时索引，返回 ingest 的 stdout。"""
        code, out, err = self._call(["ingest", "data/knowledge_base", "--store", str(self.store), "--rebuild"])
        self.assertEqual(code, 0, err)
        return out

    def test_ingest_then_query_with_source(self) -> None:
        """TC-E2E-001：示例数据入库后 query 返回带来源的命中（AC2）。"""
        out = self._ingest()
        match = re.search(r"已入库：(\d+) 篇文档，(\d+) 个片段", out)
        self.assertIsNotNone(match, out)
        self.assertGreaterEqual(int(match.group(1)), 3)
        self.assertGreaterEqual(int(match.group(2)), 3)

        code, query_out, err = self._call(
            ["query", "如何把知识库接入 DeepSeek Agent", "--store", str(self.store)]
        )
        self.assertEqual(code, 0, err)
        lines = query_out.strip().splitlines()
        self.assertTrue(lines, "示例知识库检索应有命中")
        for line in lines:
            # 余弦相似度可能为负，分数允许前导负号。
            self.assertRegex(line, r"^\[\d+\] -?\d+\.\d{6} 来源：data/knowledge_base/.+$")

    def test_query_context_has_citation(self) -> None:
        """TC-E2E-002：--context 输出带 [1] 引用标记的来源上下文（AC2）。"""
        self._ingest()
        code, out, err = self._call(["query", "引用规则", "--context", "--store", str(self.store)])
        self.assertEqual(code, 0, err)
        self.assertIn("[1] 来源：data/knowledge_base/", out)
        first_block = out.split("\n\n")[0].split("\n", 1)
        self.assertTrue(first_block[0].startswith("[1] 来源："))
        self.assertTrue(first_block[1].strip())

    def test_json_query_on_sample_data(self) -> None:
        """TC-E2E 补充：示例数据上 --json 输出可解析且含 results。"""
        self._ingest()
        code, out, err = self._call(["query", "检索工具有哪些参数", "--json", "--store", str(self.store)])
        self.assertEqual(code, 0, err)
        payload = json.loads(out)
        self.assertEqual(set(payload), {"query", "results"})
        self.assertIsInstance(payload["results"], list)


if __name__ == "__main__":
    unittest.main()
