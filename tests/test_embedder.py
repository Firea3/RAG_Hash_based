"""embedder 模块测试：hashing 嵌入的算法性质与确定性、OpenAI 兼容嵌入的 Key/请求/响应校验。

对应用例：TC-EMB-001 ~ TC-EMB-010（docs/04_test_plan.md 第 5.3 节）。
网络出口 ``rag.embedder._post_json`` 全部用 unittest.mock 替换，禁止真实联网。
"""

import json
import math
import os
import subprocess
import sys
import unittest
from unittest import mock

from rag.config import RagConfig
from rag.embedder import HashingEmbedder, OpenAICompatibleEmbedder, build_embedder
from tests._support import PROJECT_ROOT


def _cosine(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class HashingEmbedderTest(unittest.TestCase):
    """HashingEmbedder 的契约测试。"""

    def test_defaults_and_shape(self) -> None:
        """TC-EMB-001：name=hashing、dim=256，输出形状与元素类型正确。"""
        embedder = HashingEmbedder()
        self.assertEqual(embedder.name, "hashing")
        self.assertEqual(embedder.dim, 256)
        vectors = embedder.embed(["你好", "world"])
        self.assertEqual(len(vectors), 2)
        for vector in vectors:
            self.assertEqual(len(vector), 256)
            self.assertTrue(all(isinstance(value, float) for value in vector))

    def test_l2_normalized_and_empty_text_zero(self) -> None:
        """TC-EMB-002：非空文本向量 L2 范数为 1；空文本返回全 0 向量。"""
        vectors = HashingEmbedder().embed(["一段正常文本", ""])
        first_norm = math.sqrt(sum(value * value for value in vectors[0]))
        self.assertAlmostEqual(first_norm, 1.0, places=6)
        self.assertEqual(vectors[1], [0.0] * 256)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in vectors[1])), 0.0)

    def test_empty_list_returns_empty(self) -> None:
        """TC-EMB-003：embed([]) 返回 []。"""
        self.assertEqual(HashingEmbedder().embed([]), [])

    def test_cross_process_deterministic(self) -> None:
        """TC-EMB-004：两个独立进程对同一文本得到完全相同向量（跨进程确定性，未用内置 hash）。"""
        script = (
            "import json;"
            "from rag.embedder import HashingEmbedder;"
            "print(json.dumps(HashingEmbedder().embed(['测试文本'])))"
        )
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        outputs = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                env=env,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1])

    def test_similar_text_scores_higher(self) -> None:
        """TC-EMB-005：语义相近文本的余弦相似度高于无关文本。"""
        embedder = HashingEmbedder()
        base = embedder.embed(["知识库检索"])[0]
        similar = embedder.embed(["检索知识库"])[0]
        unrelated = embedder.embed(["天气预报明天有雨"])[0]
        self.assertGreater(_cosine(base, similar), _cosine(base, unrelated))

    def test_custom_dim(self) -> None:
        """TC-EMB-010：自定义 dim 生效，向量维度严格等于 dim。"""
        vectors = HashingEmbedder(dim=8).embed(["文本"])
        self.assertEqual(len(vectors[0]), 8)


class OpenAICompatibleEmbedderTest(unittest.TestCase):
    """OpenAICompatibleEmbedder 的契约测试（全部走 mock，不联网）。"""

    DIM = 8

    def _make(self, api_key: str | None = None, **kwargs) -> OpenAICompatibleEmbedder:
        """构造小维度嵌入器，便于断言。"""
        return OpenAICompatibleEmbedder(
            "http://example.invalid", "m", api_key=api_key, dim=self.DIM, **kwargs
        )

    @staticmethod
    def _response(count: int, dim: int) -> dict:
        """构造合法响应。"""
        return {"data": [{"embedding": [0.1] * dim} for _ in range(count)]}

    def test_api_key_priority(self) -> None:
        """TC-EMB-006：Key 优先级 传入 > RAG_API_KEY > OPENAI_API_KEY；都没有则 RuntimeError 且不发起请求。"""
        with mock.patch("rag.embedder._post_json", return_value=self._response(1, self.DIM)) as post:
            with mock.patch.dict(os.environ, {"RAG_API_KEY": "env1", "OPENAI_API_KEY": "env2"}, clear=True):
                self._make().embed(["q"])
                self.assertEqual(post.call_args.args[2]["Authorization"], "Bearer env1")
                self._make(api_key="explicit").embed(["q"])
                self.assertEqual(post.call_args.args[2]["Authorization"], "Bearer explicit")
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env2"}, clear=True):
                self._make().embed(["q"])
                self.assertEqual(post.call_args.args[2]["Authorization"], "Bearer env2")
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(RuntimeError):
                    self._make().embed(["q"])
                self.assertEqual(post.call_count, 3)

    def test_request_shape(self) -> None:
        """TC-EMB-007：URL、请求体、请求头与超时符合契约。"""
        with mock.patch(
            "rag.embedder._post_json",
            return_value=self._response(2, self.DIM),
        ) as post:
            embedder = OpenAICompatibleEmbedder(
                "http://example.invalid/", "m", api_key="k", dim=self.DIM, timeout=12.5
            )
            vectors = embedder.embed(["q1", "q2"])
        self.assertEqual(post.call_count, 1)
        url, payload, headers, timeout = post.call_args.args
        self.assertEqual(url, "http://example.invalid/embeddings")
        self.assertEqual(payload, {"model": "m", "input": ["q1", "q2"]})
        self.assertEqual(headers["Authorization"], "Bearer k")
        self.assertIn("Content-Type", headers)
        self.assertEqual(timeout, 12.5)
        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(vectors[0]), self.DIM)

    def test_bad_response_raises_runtime_error(self) -> None:
        """TC-EMB-008：维度不符、结构缺失、空 data 三种响应都抛 RuntimeError。"""
        bad_responses = [
            {"data": [{"embedding": [0.1] * 10}]},
            {},
            {"data": []},
        ]
        for payload in bad_responses:
            with self.subTest(payload=payload):
                with mock.patch("rag.embedder._post_json", return_value=payload):
                    with self.assertRaises(RuntimeError):
                        self._make().embed(["q"])

    def test_returns_vectors_and_respects_dim(self) -> None:
        """TC-EMB-010：合法响应按 data[i]["embedding"] 解析，维度由配置决定。"""
        with mock.patch("rag.embedder._post_json", return_value=self._response(1, 16)):
            vectors = OpenAICompatibleEmbedder("http://example.invalid", "m", api_key="k", dim=16).embed(["q"])
        self.assertEqual(len(vectors[0]), 16)
        with mock.patch("rag.embedder._post_json", return_value=self._response(1, 8)):
            with self.assertRaises(RuntimeError):
                OpenAICompatibleEmbedder("http://example.invalid", "m", api_key="k", dim=16).embed(["q"])


class BuildEmbedderTest(unittest.TestCase):
    """build_embedder 分派测试。"""

    def test_build_embedder_branches(self) -> None:
        """TC-EMB-009：hashing / openai / 其他值三分支。"""
        hashing = build_embedder(RagConfig())
        self.assertIsInstance(hashing, HashingEmbedder)
        self.assertEqual(hashing.dim, 256)

        openai_embedder = build_embedder(RagConfig(embedder="openai", api_key="k", dim=8))
        self.assertIsInstance(openai_embedder, OpenAICompatibleEmbedder)
        self.assertEqual(openai_embedder.name, "openai")
        self.assertEqual(openai_embedder.dim, 8)

        with self.assertRaises(ValueError):
            build_embedder(RagConfig(embedder="bad"))


if __name__ == "__main__":
    unittest.main()
