from dataclasses import dataclass, field

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.plot import LINEAR_NAMES, LOSS_YMARGIN, MODEL_NAMES, SCHEDULE_COLORS, get_figsize, get_panel_width, plot_bars, save_figure, select

# (schedule, arm) 순서대로 막대
SERIES = [("all", "random"), ("all", "oracle"), ("sequential_rescore", "random"), ("sequential_rescore", "oracle")]


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    layers: list[int] = field(default_factory=lambda: [10, 20])
    linear_name: str = "self_attn.q_proj"
    num_rotations: int = 8
    # 총 step 수, sequential_rescore는 rotation마다 total_steps / num_rotations씩 학습
    total_steps: list[int] = field(default_factory=lambda: [200, 1600])


def main() -> None:
    config = parse(Config)
    # paro_pair에서 낸 결과라 rotation 결과와 섞지 않고 그 안에서만 비교, random은 seed 하나
    rows = select(load_results("training_budget"), model_name=config.model_name, linear_name=config.linear_name, alpha_mode="joint", num_rotations=config.num_rotations)

    cases = [f"총 step {total_steps}" for total_steps in config.total_steps]
    fig, axes = plt.subplots(1, len(config.layers), figsize=get_figsize([get_panel_width(len(cases), len(SERIES))] * len(config.layers), 1), sharey=True, layout="constrained")
    losses = []
    for ax, layer in zip(axes, config.layers):
        bars = []
        for total_steps in config.total_steps:
            case_losses = [select(rows, layer=layer, schedule=schedule, arm=arm, total_steps=total_steps)[0]["loss"] for schedule, arm in SERIES]
            losses += case_losses
            bars.append([(loss, SCHEDULE_COLORS[series]) for loss, series in zip(case_losses, SERIES)])
        plot_bars(ax, cases, bars, "%")
        ax.set_title(f"layer {layer}")
        ax.set_ylabel("loss")
    # 막대 사이 차이가 보이도록 가장 낮은 막대 조금 아래에서 y축을 자름
    axes[0].set_ylim(bottom=min(losses) - LOSS_YMARGIN)

    fig.suptitle(f"{MODEL_NAMES[config.model_name]} {LINEAR_NAMES[config.linear_name]} rotation {config.num_rotations}개 rotation마다 다시 고르기")
    fig.legend(handles=[Patch(color=SCHEDULE_COLORS[series], label=f"{series[0]} {series[1]}") for series in SERIES], loc="outside lower center", ncol=len(SERIES))
    save_figure(fig, "rescore_budget")


if __name__ == "__main__":
    main()
