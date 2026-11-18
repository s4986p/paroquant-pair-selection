from dataclasses import dataclass, field
from math import sqrt
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.plot import LINEAR_NAMES, MODEL_NAMES, get_figsize, get_panel_width, plot_bars, save_figure, select

# openreview 답변 q4 표의 저자 값, scaling은 alpha만, rotation 8개는 그 위에 rotation 8개
AUTHOR_LOSSES = {"self_attn.q_proj": (0.78, 0.53), "mlp.gate_proj": (0.75, 0.55)}
METRIC_COLORS = {"저자": "#a3a29b", "l2": "#eb6834", "mse": "#2a78d6"}


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    layer: int = 0
    linear_names: list[str] = field(default_factory=lambda: list(AUTHOR_LOSSES))
    num_rotations: int = 8
    num_steps: int = 200


def main() -> None:
    config = parse(Config)
    rows = select(load_results("rotation"), model_name=config.model_name, layer=config.layer, num_steps=config.num_steps)

    cases = ["scaling", f"rotation {config.num_rotations}개"]
    panel_width = get_panel_width(len(cases), len(METRIC_COLORS))
    fig, axes = plt.subplots(1, len(config.linear_names), figsize=get_figsize([panel_width] * len(config.linear_names), 1), squeeze=False, layout="constrained")
    for ax, linear_name in zip(axes[0], config.linear_names):
        author_scaling, author_rotation = AUTHOR_LOSSES[linear_name]
        scaling = select(rows, linear_name=linear_name, arm="scaling")[0]["loss"]
        random_losses = [row["loss"] for row in select(rows, linear_name=linear_name, arm="random", alpha_mode="joint", schedule="all", num_rotations=config.num_rotations)]
        # loss는 mse 비율이라, 저자 표기인 l2로 재면 그 제곱근, random은 seed 평균
        bars = [
            [(author_scaling, METRIC_COLORS["저자"]), (sqrt(scaling), METRIC_COLORS["l2"]), (scaling, METRIC_COLORS["mse"])],
            [(author_rotation, METRIC_COLORS["저자"]), (mean(random_losses), METRIC_COLORS["mse"])],
        ]
        plot_bars(ax, cases, bars, "%")
        ax.set_ylim(bottom=0)
        ax.set_title(LINEAR_NAMES[linear_name])
        ax.set_ylabel("loss")

    fig.suptitle(f"{MODEL_NAMES[config.model_name]} layer {config.layer} 저자 표 재현")
    handles = [Patch(color=color, label=name) for name, color in METRIC_COLORS.items()]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    save_figure(fig, "reproduction")


if __name__ == "__main__":
    main()
