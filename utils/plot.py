from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

MODEL_NAMES = {"Qwen3-0.6B": "qwen3-0.6b"}
LINEAR_NAMES = {"self_attn.q_proj": "q_proj", "mlp.gate_proj": "gate_proj"}
BAR_INCHES = 0.45
BAR_GAP = 1
PANEL_HEIGHT = 3.0
PANEL_MARGIN = 0.9
# case가 적어도 제목과 범례가 잘리지 않는 너비
MIN_FIGURE_WIDTH = 9.0
# 막대 위 값이 그림 밖으로 잘리지 않게 y축 위 여백 비율
VALUE_MARGIN = 0.15
# 막대 위 값 글자 크기와 막대에서 띄우는 point 수
VALUE_FONTSIZE = 8
VALUE_OFFSET = 3

plt.rcParams.update({
    "font.family": "Apple SD Gothic Neo",
    "axes.unicode_minus": False,
    "figure.titlesize": 13,
    "figure.constrained_layout.w_pad": 0.08,
    "figure.constrained_layout.h_pad": 0.08,
    "figure.constrained_layout.wspace": 0.08,
    "figure.constrained_layout.hspace": 0.08,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#e8e7e3",
    "grid.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "savefig.dpi": 200,
})


def get_panel_width(
    num_cases: int,
    num_series: int,
) -> float:
    # case마다 series 막대와 간격
    return BAR_INCHES * num_cases * (num_series + BAR_GAP)


def get_figsize(
    panel_widths: list[float],
    num_rows: int,
) -> tuple[float, float]:
    # panel을 가로로, 묶음을 세로로 놓음
    return max(sum(panel_widths) + PANEL_MARGIN * len(panel_widths), MIN_FIGURE_WIDTH), PANEL_HEIGHT * num_rows + 1


def plot_bars(
    ax: Axes,
    cases: list[str],
    bars: list[list[tuple[float, str]]],
) -> None:
    # case마다 막대를 나란히 그리고 막대 위에 값 표시, case마다 막대 수가 달라도 가운데 정렬
    # bars는 case마다 (값, 색) 목록
    # 예시) cases ["scaling", "rotation 8개"], bars [[(0.78, 회색), (0.76, 파랑)], [(0.53, 파랑)]]
    case_width = max(len(case_bars) for case_bars in bars) + BAR_GAP
    for case_index, case_bars in enumerate(bars):
        for bar_index, (value, color) in enumerate(case_bars):
            x = case_index * case_width + bar_index - (len(case_bars) - 1) / 2
            ax.bar(x, value, width=1, color=color)
            ax.annotate(f"{value:.1%}", (x, value), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    ax.set_xticks([case_index * case_width for case_index in range(len(cases))], cases)
    ax.set_xlim(-case_width / 2, (len(cases) - 0.5) * case_width)
    ax.set_ylim(bottom=0)
    ax.margins(y=VALUE_MARGIN)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))


def save_figure(
    fig: Figure,
    file_name: str,
) -> None:
    path = Path(f"figures/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
