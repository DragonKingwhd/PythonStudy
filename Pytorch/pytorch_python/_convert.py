"""把 Inside-Deep-Learning 仓库的 .ipynb 转成可直接 `python` 运行的 .py 脚本。

用法:
    python _convert.py            # 转换全部 Chapter_*.ipynb
    python _convert.py 1 3 5      # 只转换 Chapter_1/3/5
"""
import ast
import json
import re
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "Inside-Deep-Learning"
DST_DIR = Path(__file__).resolve().parent

# 需要剥离的 Colab / IPython 专属行（整行改成注释保留痕迹）
STRIP_LINE_PATTERNS = [
    re.compile(r"^\s*%"),                          # %matplotlib / %%timeit 等 magic
    re.compile(r"^\s*!"),                          # !pip install xxx
    re.compile(r"^\s*from\s+google\.colab\b"),
    re.compile(r"^\s*import\s+google\.colab\b"),
    re.compile(r"^\s*drive\.mount\b"),
    re.compile(r"^\s*from\s+IPython\.display\s+import\s+set_matplotlib_formats"),
    re.compile(r"^\s*set_matplotlib_formats\("),
]

# 单行/全局替换
REPLACE_PATTERNS = [
    (re.compile(r"\btqdm\.autonotebook\b"), "tqdm"),
    # CUDA 自动回退到 CPU
    (re.compile(r'torch\.device\(\s*[\'"]cuda[\'"]\s*\)'),
     'torch.device("cuda" if torch.cuda.is_available() else "cpu")'),
]


def clean_lines(src: str) -> str:
    out = []
    for ln in src.splitlines():
        if any(p.search(ln) for p in STRIP_LINE_PATTERNS):
            out.append("# [stripped] " + ln.strip())
            continue
        for pat, repl in REPLACE_PATTERNS:
            ln = pat.sub(repl, ln)
        out.append(ln)
    return "\n".join(out)


def wrap_trailing_expr(code: str) -> str:
    """Jupyter 里 cell 末尾的裸表达式会被自动 display；在 .py 里要 print 出来才有效。"""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body:
        return code
    last = tree.body[-1]
    if not isinstance(last, ast.Expr):
        return code
    # 跳过已经是 print/plt.xxx/sns.xxx/display 的调用 —— 它们本身有副作用
    val = last.value
    if isinstance(val, ast.Call):
        func = val.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
            # 取 root
            root = func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {"plt", "sns", "ax", "fig", "display"}:
                return code
        if name in {"print", "display", "show"}:
            return code

    lines = code.splitlines()
    start = last.lineno - 1
    end = (getattr(last, "end_lineno", last.lineno) or last.lineno) - 1
    indent_match = re.match(r"\s*", lines[start])
    indent = indent_match.group(0) if indent_match else ""
    # 用 ast.unparse 拿到不含注释的表达式文本，避免行内 # 注释把 print(...) 的右括号吞掉
    try:
        body = ast.unparse(val)
    except Exception:
        return code
    new_lines = lines[:start] + [f"{indent}print({body})"] + lines[end + 1:]
    return "\n".join(new_lines)


PLOT_HINT = re.compile(r"\b(plt\.|sns\.|\.plot\(|\.imshow\(|\.hist\()")


def maybe_append_show(code: str) -> str:
    if PLOT_HINT.search(code) and "plt.show" not in code and "savefig" not in code:
        return code.rstrip() + "\nplt.show()"
    return code


def md_to_comment(src: str) -> str:
    lines = src.splitlines() or [""]
    return "\n".join("# " + ln if ln else "#" for ln in lines)


def convert(nb_path: Path, out_path: Path) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    chunks = [
        f'"""{nb_path.stem} — 来自 Inside-Deep-Learning 仓库，自动转成可运行 .py 脚本。',
        "",
        "原始 notebook 位于 ../Inside-Deep-Learning/。",
        '"""',
        "",
    ]
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell["source"]).rstrip()
        if not src:
            continue
        if cell["cell_type"] == "markdown":
            chunks.append(f"# ====== Cell {i} (markdown) ======")
            chunks.append(md_to_comment(src))
        else:
            chunks.append(f"# ====== Cell {i} (code) ======")
            cleaned = clean_lines(src)
            cleaned = wrap_trailing_expr(cleaned)
            cleaned = maybe_append_show(cleaned)
            chunks.append(cleaned)
        chunks.append("")
    out_path.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    print(f"  -> {out_path.name} ({out_path.stat().st_size} bytes)")


def main(argv: list[str]) -> None:
    if argv:
        targets = [SRC_DIR / f"Chapter_{n}.ipynb" for n in argv]
    else:
        targets = sorted(SRC_DIR.glob("Chapter_*.ipynb"),
                         key=lambda p: int(re.search(r"\d+", p.stem).group()))
    for nb in targets:
        if not nb.exists():
            print(f"skip (not found): {nb}")
            continue
        out = DST_DIR / (nb.stem.lower() + ".py")
        print(f"converting {nb.name}")
        convert(nb, out)


if __name__ == "__main__":
    main(sys.argv[1:])
