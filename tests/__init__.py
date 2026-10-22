"""自动化测试包。

存在原因：契约要求用 ``python -m unittest discover -s tests -t .`` 运行，
当起始目录与顶层目录不同时，unittest 要求起始目录是一个包（含 __init__.py），
否则会报 ``ImportError: Start directory is not importable``。
"""
