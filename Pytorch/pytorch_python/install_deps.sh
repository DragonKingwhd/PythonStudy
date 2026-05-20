#!/usr/bin/env bash
# 在 da_server conda 环境里补装 Inside-Deep-Learning 14 章脚本缺少的依赖。
# 用法:
#   bash install_deps.sh
# 或激活后:
#   conda activate da_server && bash install_deps.sh

set -e

PY="/home/user/miniconda3/envs/da_server/bin/python"
PIP="$PY -m pip"

if [ ! -x "$PY" ]; then
    echo "未找到 da_server 环境的 python: $PY"
    exit 1
fi

echo "==> 使用解释器: $($PY --version) ($PY)"

# 先升级 pip（避免老版本解析依赖失败）
$PIP install --upgrade pip

# 这些是 14 章脚本里 import 到、但 da_server 环境缺失的第三方包。
# torch / torchvision / numpy / scipy / Pillow / imageio / tqdm 已经安装，跳过。
PKGS=(
    "ipython"          # chapter_5/7/8/11/12 用到 IPython.display
    "matplotlib"       # 所有章节都画图
    "seaborn"          # 所有章节都画图
    "pandas"           # 数据处理
    "pytz"             # pandas 依赖（环境里缺）
    "python-dateutil"  # pandas 依赖
    "idna"             # requests 依赖（环境里缺）
    "requests"         # 下载数据集
    "scikit-learn"     # MNIST 等示例数据
    "optuna"           # chapter_5 超参搜索
    "transformers"     # chapter_13 BERT
)

echo "==> 安装通用依赖..."
$PIP install "${PKGS[@]}"

# torchtext 跟 torch 版本强绑定，单独处理。
# da_server 当前是 torch 2.11+cu128，PyPI 上的 torchtext 0.18 只支持到 torch 2.3，
# 直接装会报版本不兼容。这里先用 --no-deps 装上可用最新版以满足 import；
# 如果实际跑 chapter_12/13 还报错，再考虑用 torchtext 自带的 legacy 替代品（datasets/transformers tokenizer）。
echo "==> 尝试安装 torchtext（与 torch 版本不一定完全匹配，按需）..."
$PIP install --no-deps torchtext || echo "torchtext 安装失败，可后续手动处理"

echo "==> 完成。复核结果："
$PY - << 'PYEOF'
import importlib
mods = ["IPython","PIL","imageio","matplotlib","numpy","optuna","pandas",
        "requests","scipy","seaborn","sklearn","torch","torchtext",
        "torchvision","tqdm","transformers"]
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  [OK]  {m}")
    except Exception as e:
        print(f"  [FAIL] {m}: {str(e).splitlines()[0]}")
PYEOF
