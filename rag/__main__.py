"""python -m rag 入口。"""

# 从命令行模块导入 main 函数：真正解析参数、执行子命令的逻辑在 rag/cli.py 里。
from rag.cli import main

# 只有直接运行这个文件时才执行下面的代码；被别的文件 import 时不会自动运行，像“仅限本店使用”的开关。
if __name__ == "__main__":
    # 调用 main() 并把返回码交给操作系统：0 表示成功、1 表示出错，像交卷时把分数报给老师。
    raise SystemExit(main())
