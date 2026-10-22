"""cli 模块测试：三个子命令、stdout 格式、stderr 错误、退出码 0/1/2、python -m rag 入口。

对应用例：TC-CLI-001 ~ TC-CLI-012（docs/04_test_plan.md 第 5.7 节）。
所有索引写入临时目录；子进程用例只调用不写盘、不联网的子命令。
"""

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

from rag.cli import main
from rag.tool import TOOL_SPEC
from tests._support import PROJECT_ROOT, temp_dir


class CliTest(unittest.TestCase):
    """rag.cli.main 的契约测试。"""

    def setUp(self) -> None:
        self.root = temp_dir(self)
        self.store = self.root / "index.json"
        self.kb = self.root / "kb"
        self.kb.mkdir()
        (self.kb / "a.md").write_text("知识库检索需要先切片再计算向量。" * 5, encoding="utf-8")
        (self.kb / "b.md").write_text("引用规则：每个结论后用编号标注来源。" * 5, encoding="utf-8")

    def _call(self, argv):
        """调用 main 并捕获 stdout/stderr，返回 (退出码, stdout, stderr)。"""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def _call_expect_exit(self, argv):
        """调用 main 并捕获 SystemExit，返回 (退出码, stdout, stderr)。"""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                main(argv)
        return ctx.exception.code, out.getvalue(), err.getvalue()

    def _ingest(self, *extra):
        """先入库，返回 (退出码, stdout, stderr)。"""
        return self._call(["ingest", str(self.kb), "--store", str(self.store), *extra])

    def test_ingest_success_output_format(self) -> None:
        """TC-CLI-001：ingest 成功打印固定格式并返回 0，索引文件落盘。"""
        code, out, err = self._ingest()
        self.assertEqual(code, 0)
        pattern = rf"已入库：\d+ 篇文档，\d+ 个片段 -> {re.escape(str(self.store))}"
        self.assertRegex(out.strip(), pattern)
        self.assertEqual(err, "")
        self.assertTrue(self.store.exists())

    def test_query_default_output_lines(self) -> None:
        """TC-CLI-002：query 默认逐行打印 [n] 分数 来源：路径，返回 0。"""
        self._ingest()
        code, out, err = self._call(["query", "知识库检索", "--store", str(self.store)])
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        lines = out.strip().splitlines()
        self.assertTrue(lines)
        for line in lines:
            # 余弦相似度可能为负（hashing 嵌入允许负分），故分数允许前导负号。
            self.assertRegex(line, r"^\[\d+\] -?\d+\.\d{6} 来源：.+$")

    def test_query_no_hits_prints_message(self) -> None:
        """TC-CLI-003：无命中时打印“未找到相关内容。”并返回 0。"""
        code, out, err = self._call(["query", "完全无关的问题", "--store", str(self.root / "empty.json")])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "未找到相关内容。")
        self.assertEqual(err, "")

    def test_query_json_output(self) -> None:
        """TC-CLI-004：--json 输出与 run_tool 相同的 JSON 结构，中文不转义。"""
        self._ingest()
        code, out, _ = self._call(["query", "知识库检索", "--json", "--store", str(self.store)])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(set(payload), {"query", "results"})
        self.assertEqual(payload["query"], "知识库检索")
        self.assertIn("知识库检索", out)
        self.assertNotIn("\\u", out)

    def test_query_context_output(self) -> None:
        """TC-CLI-005：--context 输出带 [n] 引用的上下文；无命中时无输出（实现定义）。"""
        self._ingest()
        code, out, _ = self._call(["query", "知识库检索", "--context", "--store", str(self.store)])
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("[1] 来源："))
        self.assertRegex(out.splitlines()[0], r"^\[1\] 来源：.+$")

        code2, out2, _ = self._call(["query", "完全无关", "--context", "--store", str(self.root / "empty.json")])
        self.assertEqual(code2, 0)
        self.assertEqual(out2, "")

    def test_tool_schema_output(self) -> None:
        """TC-CLI-006：tool-schema 打印 TOOL_SPEC 的 JSON 并返回 0。"""
        code, out, err = self._call(["tool-schema"])
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(json.loads(out), TOOL_SPEC)

    def test_runtime_error_returns_1(self) -> None:
        """TC-CLI-007：运行期错误 stderr 打印“错误：”并返回 1，不创建索引文件。"""
        missing_store = self.root / "never.json"
        code, out, err = self._call(["ingest", str(self.root / "nope"), "--store", str(missing_store)])
        self.assertEqual(code, 1)
        self.assertEqual(out, "")
        self.assertTrue(err.startswith("错误："), err)
        self.assertFalse(missing_store.exists())

        # 缺 API Key 的 OpenAI 嵌入：运行期错误 → 退出码 1，且绝不联网。
        with mock.patch.dict(os.environ, {}, clear=True):
            code2, out2, err2 = self._call(
                ["query", "问题", "--store", str(self.root / "empty2.json"), "--embedder", "openai"]
            )
        self.assertEqual(code2, 1)
        self.assertTrue(err2.startswith("错误："), err2)

    def test_argparse_usage_error_exit_2(self) -> None:
        """TC-CLI-008：argparse 用法错误（无子命令、未知子命令、缺必填参数）退出码为 2。"""
        for argv in ([], ["unknown-cmd"], ["query"]):
            with self.subTest(argv=argv):
                code, _, _ = self._call_expect_exit(argv)
                self.assertEqual(code, 2)

    def test_parameters_take_effect(self) -> None:
        """TC-CLI-009：--store、--dim、--top-k 生效。"""
        code, _, _ = self._ingest("--dim", "8")
        self.assertEqual(code, 0)
        payload = json.loads(self.store.read_text(encoding="utf-8"))
        self.assertEqual(payload["embedder"]["dim"], 8)
        code2, out2, _ = self._call(["query", "知识库检索", "--top-k", "1", "--dim", "8", "--store", str(self.store)])
        self.assertEqual(code2, 0)
        self.assertLessEqual(len(out2.strip().splitlines()), 1)

    def test_main_is_reentrant(self) -> None:
        """TC-CLI-010：main(argv) 无全局状态污染，可重复调用且输出一致。"""
        code1, out1, _ = self._call(["tool-schema"])
        code2, out2, _ = self._call(["tool-schema"])
        self.assertEqual((code1, code2), (0, 0))
        self.assertEqual(out1, out2)


class CliSubprocessTest(unittest.TestCase):
    """python -m rag 进程级入口测试（离线、只读子命令）。"""

    def _run(self, args):
        """在项目根目录用当前解释器运行 python -m rag。"""
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run(
            [sys.executable, "-m", "rag", *args],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env=env,
            check=False,
        )

    def test_python_m_rag_tool_schema(self) -> None:
        """TC-CLI-011：python -m rag tool-schema 返回 0 且输出与 TOOL_SPEC 语义相等。"""
        result = self._run(["tool-schema"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), TOOL_SPEC)

    def test_python_m_rag_no_args_exit_2(self) -> None:
        """TC-CLI-012：python -m rag 无参数时由 argparse 返回退出码 2。"""
        result = self._run([])
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
