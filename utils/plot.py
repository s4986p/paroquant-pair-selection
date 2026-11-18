from collections import defaultdict
from pathlib import Path
from statistics import mean

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

MODEL_NAMES = {"Qwen3-0.6B": "qwen3-0.6b", "Llama-3.1-8B": "llama-3.1-8b", "Mistral-7B-v0.3": "mistral-7b-v0.3", "gemma-2-9b": "gemma-2-9b"}
# random은 기준이라 회색
ARM_COLORS = {
    "random": "#a3a29b",
    "butterfly": "#6b6a66",
    "magnitude_difference": "#eb6834",
    "magnitude_similarity": "#b5651d",
    "correlation": "#1baf7a",
    "activation_magnitude_difference": "#d62728",
    "oracle": "#8e5ccf",
}
LINEAR_NAMES = {"self_attn.q_proj": "q_proj", "mlp.gate_proj": "gate_proj"}
BAR_INCHES = 0.45
BAR_GAP = 1
PANEL_HEIGHT = 3.0
PANEL_MARGIN = 0.9
# case가 적어도 제목과 범례가 잘리지 않는 너비
MIN_FIGURE_WIDTH = 9.0
# 막대 위 값이 그림 밖으로 잘리지 않게 y축 위 여백 비율
VALUE_MARGIN = 0.15
# rotation마다 다시 고르기 plot의 (schedule, arm) 색, all은 arm 색, sequential_rescore는 더 진한 색
SCHEDULE_COLORS = {
    ("all", "random"): ARM_COLORS["random"],
    ("all", "oracle"): ARM_COLORS["oracle"],
    ("sequential_rescore", "random"): "#5b5a56",
    ("sequential_rescore", "oracle"): "#5b2a9e",
}
# 이름이 길어 눈금에서는 _ 자리에서 줄을 바꿈
ARM_LABELS = {
    "random": "random",
    "butterfly": "butterfly",
    "magnitude_difference": "magnitude\ndifference",
    "magnitude_similarity": "magnitude\nsimilarity",
    "correlation": "correlation",
    "activation_magnitude_difference": "activation\nmagnitude\ndifference",
    "oracle": "oracle",
}
# %로 그리는 unit, 나머지는 format_value에서 개수와 소수로 표시
PERCENT_UNITS = ["%", "%p"]
# 기준선 회색
NEUTRAL_COLOR = "#808080"
BASELINE_WIDTH = 0.8
# arm 사이 차이가 보이도록 가장 낮은 막대보다 이만큼 아래에서 y축을 자름
LOSS_YMARGIN = 0.04
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


def select(
    rows: list[dict],
    **conditions: object,
) -> list[dict]:
    # 조건이 list면 그중 하나, 경로에 없는 설정은 None으로 봄
    # 예시) select(rows, layer=[10, 20], arm="random")
    return [row for row in rows if all(row.get(name) in (value if isinstance(value, list) else [value]) for name, value in conditions.items())]


def compute_means(
    rows: list[dict],
    metric: str,
    cell_names: tuple[str, ...],
) -> dict[str, float]:
    # random은 seed마다 값이 달라 cell마다 seed 평균을 먼저 내고, arm마다 cell 평균을 냄
    # 예시) cell_names ("model_name",) → arm마다 모델 4개 평균
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        values[row["arm"]][tuple(row[name] for name in cell_names)].append(row[metric])
    return {arm: mean(mean(cell_values) for cell_values in arm_values.values()) for arm, arm_values in values.items()}


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


def format_value(
    value: float,
    unit: str,
) -> str:
    # 예시) "%" → "53.0%", "%p" → "+2.2%p", "개" → "2.6개", "" → "0.64"
    # 반올림해 0이 된 음수가 "-0.0"으로 표시되지 않게 0.0을 더함
    if unit == "%p":
        return f"{round(value * 100, 1) + 0.0:+.1f}%p"
    if unit == "%":
        return f"{value:.1%}"
    if unit == "개":
        return f"{value:.1f}개"
    return f"{value:.2f}"


def plot_bars(
    ax: Axes,
    cases: list[str],
    bars: list[list[tuple[float, str]]],
    unit: str,
) -> None:
    # case마다 막대를 나란히 그리고 막대 위에 format_value로 값 표시, case마다 막대 수가 달라도 가운데 정렬
    # bars는 case마다 (값, 색) 목록
    # 예시) cases ["scaling", "rotation 8개"], bars [[(0.78, 회색), (0.76, 파랑)], [(0.53, 파랑)]]
    case_width = max(len(case_bars) for case_bars in bars) + BAR_GAP
    for case_index, case_bars in enumerate(bars):
        for bar_index, (value, color) in enumerate(case_bars):
            x = case_index * case_width + bar_index - (len(case_bars) - 1) / 2
            ax.bar(x, value, width=1, color=color)
            ax.annotate(format_value(value, unit), (x, max(value, 0)), xytext=(0, VALUE_OFFSET), textcoords="offset points", ha="center", va="bottom", fontsize=VALUE_FONTSIZE)
    ax.axhline(0, color=NEUTRAL_COLOR, linewidth=BASELINE_WIDTH)
    ax.set_xticks([case_index * case_width for case_index in range(len(cases))], cases)
    ax.set_xlim(-case_width / 2, (len(cases) - 0.5) * case_width)
    ax.margins(y=VALUE_MARGIN)
    # %p는 눈금 간격이 1%p보다 작을 수 있어 소수 첫째 자리까지
    if unit in PERCENT_UNITS:
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1 if unit == "%p" else 0, symbol=unit))


def add_arm_legend(
    fig: Figure,
    arms: list[str],
) -> None:
    fig.legend(handles=[Patch(color=ARM_COLORS[arm], label=arm) for arm in arms], loc="outside lower center", ncol=len(arms))


def save_figure(
    fig: Figure,
    file_name: str,
) -> None:
    path = Path(f"figures/{file_name}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"saved {path}")
