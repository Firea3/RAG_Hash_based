"""测试公共辅助模块（tests 包内部使用，不含测试用例）。

环境适配说明（2026-10-22 在本机实测）
------------------------------------
本机 Windows 沙箱下，用 ``mode=0o700`` 创建的目录对受限令牌不可读写：
``tempfile.TemporaryDirectory()`` / ``tempfile.mkdtemp()`` 创建目录后立即
抛 ``PermissionError``，且该目录连删除都做不到（连 ``icacls`` 也被拒绝）。
默认模式（0o755/0o777）创建的目录则完全正常，嵌套目录与清理也已实测通过。

因此临时目录统一改用 ``tempfile.gettempdir()`` 下的默认模式目录，
对外语义与 ``TemporaryDirectory`` 一致（用 with 或 addCleanup 清理）。
此适配只影响测试，不影响被测代码 ``rag/``。
"""

import contextlib
import io
import os
import shutil
import tempfile
import uuid
from hashlib import sha256
from pathlib import Path

# 项目根目录（tests/ 的上一级），供子进程用例与只读示例数据使用。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TempDir:
    """可读写、可清理的临时目录。

    参数：prefix 目录名前缀。
    用法：``with TempDir() as name: ...``；或 ``handle.cleanup()`` 手动清理。
    """

    def __init__(self, prefix: str = "ragtest_") -> None:
        self.name = os.path.join(tempfile.gettempdir(), prefix + uuid.uuid4().hex)
        # 默认模式创建（不要传 0o700），否则本机沙箱下目录不可访问。
        os.makedirs(self.name)

    def __enter__(self) -> str:
        return self.name

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        """删除临时目录（尽力而为，不抛异常）。参数：无；返回：无。"""
        shutil.rmtree(self.name, ignore_errors=True)


def temp_dir(test_case, prefix: str = "ragtest_") -> Path:
    """为用例创建临时目录并注册清理。

    参数：test_case 当前 TestCase（用其 addCleanup）、prefix 前缀。
    返回：临时目录的 Path。
    """
    handle = TempDir(prefix)
    test_case.addCleanup(handle.cleanup)
    return Path(handle.name)


@contextlib.contextmanager
def captured_io():
    """捕获标准输出与标准错误。

    参数：无；返回：上下文管理器，产出 (out, err) 两个 io.StringIO。
    """
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


def fingerprint(path) -> str | None:
    """计算文件内容指纹，用于证明测试没有改动真实索引。

    参数：path 文件路径；返回：sha256 十六进制串，文件不存在时返回 None。
    """
    target = Path(path)
    if not target.exists():
        return None
    return sha256(target.read_bytes()).hexdigest()
