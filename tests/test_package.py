"""包级与数据模型测试：__init__ 导出、版本、无第三方依赖、导入无副作用、models/config 契约。

对应用例：TC-PKG-001 ~ 004、TC-MODEL-001 ~ 004、TC-CFG-001 ~ 002
（docs/04_test_plan.md 第 5.8 ~ 5.10 节）。
"""

import ast
import dataclasses
import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import rag
from rag.config import RagConfig
from rag.models import Chunk, Document, Hit
from tests._support import PROJECT_ROOT, temp_dir

RAG_MODULES = [
    "rag.config",
    "rag.models",
    "rag.loaders",
    "rag.chunker",
    "rag.embedder",
    "rag.store",
    "rag.pipeline",
    "rag.tool",
    "rag.cli",
]


class PackageTest(unittest.TestCase):
    """rag/__init__.py 与包级约束测试。"""

    def test_exports(self) -> None:
        """TC-PKG-001：7 个公共符号全部导出，且 Rag 与 pipeline.Rag 是同一对象。"""
        for name in ("Rag", "RagConfig", "Document", "Chunk", "Hit", "TOOL_SPEC", "run_tool"):
            self.assertTrue(hasattr(rag, name), name)
        self.assertIs(rag.Rag, importlib.import_module("rag.pipeline").Rag)
        self.assertIn("__version__", rag.__all__)
        self.assertIsInstance(rag.TOOL_SPEC, dict)

    def test_version(self) -> None:
        """TC-PKG-002：__version__ 精确等于 0.1.0。"""
        self.assertEqual(rag.__version__, "0.1.0")

    def test_no_third_party_dependencies(self) -> None:
        """TC-PKG-003：rag/*.py 的 import 全部来自标准库或 rag 自身（AC6）。"""
        stdlib = sys.stdlib_module_names
        for path in sorted((PROJECT_ROOT / "rag").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
            for module in imported:
                root = module.split(".")[0]
                self.assertTrue(
                    root in stdlib or root == "rag",
                    f"{path.name} 引入了非标准库模块：{module}",
                )

    def test_import_has_no_side_effects(self) -> None:
        """TC-PKG-004：导入全部模块不写盘（临时 CWD 保持为空）。"""
        workdir = temp_dir(self)
        env = {
            **os.environ,
            "PYTHONPATH": str(PROJECT_ROOT),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        code = "import " + ", ".join(RAG_MODULES)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            env=env,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(list(Path(workdir).iterdir()), [])


class ModelsTest(unittest.TestCase):
    """rag/models.py 的契约测试。"""

    def test_fields_and_defaults(self) -> None:
        """TC-MODEL-001：Document/Chunk/Hit 字段、默认 metadata 与嵌套关系正确。"""
        document = Document(doc_id="d1", text="正文", source="d1.md")
        self.assertEqual((document.doc_id, document.text, document.source), ("d1", "正文", "d1.md"))
        self.assertEqual(document.metadata, {})
        chunk = Chunk(chunk_id="c1", doc_id="d1", text="片段", source="d1.md")
        self.assertEqual(chunk.metadata, {})
        hit = Hit(chunk=chunk, score=0.5)
        self.assertIs(hit.chunk, chunk)
        self.assertEqual(hit.score, 0.5)

    def test_frozen_objects(self) -> None:
        """TC-MODEL-002：三个数据类均为 frozen，赋值抛 FrozenInstanceError。"""
        document = Document(doc_id="d1", text="正文", source="d1.md")
        chunk = Chunk(chunk_id="c1", doc_id="d1", text="片段", source="d1.md")
        hit = Hit(chunk=chunk, score=0.5)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            document.text = "改"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            chunk.text = "改"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            hit.score = 1.0

    def test_metadata_default_is_independent(self) -> None:
        """TC-MODEL-003：默认 metadata 是独立字典，不共享引用。"""
        first = Document(doc_id="d1", text="a", source="a.md")
        second = Document(doc_id="d2", text="b", source="b.md")
        first.metadata["a"] = 1
        self.assertEqual(second.metadata, {})

    def test_hit_convenience_properties(self) -> None:
        """TC-MODEL-004：Hit 的 text/source/chunk_id 直接转发到 chunk。"""
        chunk = Chunk(chunk_id="c1", doc_id="d1", text="片段正文", source="d1.md")
        hit = Hit(chunk=chunk, score=0.75)
        self.assertEqual(hit.text, chunk.text)
        self.assertEqual(hit.source, chunk.source)
        self.assertEqual(hit.chunk_id, chunk.chunk_id)


class ConfigTest(unittest.TestCase):
    """rag/config.py 的契约测试。"""

    def test_default_values(self) -> None:
        """TC-CFG-001：RagConfig 默认值逐字段对齐契约 §6.2。"""
        config = RagConfig()
        self.assertEqual(config.chunk_size, 420)
        self.assertEqual(config.chunk_overlap, 80)
        self.assertEqual(config.top_k, 4)
        self.assertEqual(config.store_path, "data/index.json")
        self.assertEqual(config.embedder, "hashing")
        self.assertEqual(config.dim, 256)
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.embed_model, "deepseek-embedding")
        self.assertIsNone(config.api_key)

    def test_override_and_frozen(self) -> None:
        """TC-CFG-002：字段可覆盖，但实例不可变（frozen=True）。"""
        config = RagConfig(chunk_size=100, top_k=2, api_key="k")
        self.assertEqual(config.chunk_size, 100)
        self.assertEqual(config.top_k, 2)
        self.assertEqual(config.api_key, "k")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.top_k = 5


if __name__ == "__main__":
    unittest.main()
