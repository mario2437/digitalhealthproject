"""논문 그림 공통 스타일. 모든 팀 스크립트에서 `import plotstyle` 한 줄로 적용한다."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

FONT_DIR = Path.home() / ".local/share/fonts"
for f in FONT_DIR.glob("NanumGothic-*.ttf"):
    fm.fontManager.addfont(str(f))

plt.rcParams.update({
    "font.family": "NanumGothic",
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 2단 논문 한 단 폭(약 9cm)에 맞춘 기본 크기 (인치)
COL_W = 3.4
PALETTE = ["#1f5e8c", "#a8511f", "#0e6a63", "#7a3b62", "#5c6873"]
