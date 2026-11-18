from dataclasses import dataclass, field
from statistics import mean

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.plot import LINEAR_NAMES, LOSS_YMARGIN, MODEL_NAMES, SCHEDULE_COLORS, compute_means, get_figsize, get_panel_width, plot_bars, save_figure, select

# (schedule, arm) 순서대로 막대, 첫 번째가 현행 구현
SERIES = [("all", "random"), ("sequential_rescore", "random"), ("sequential_rescore", "oracle")]


@dataclass
class Config:
    model_names: list[str] = field(default_factory=lambda: list(MODEL_NAMES))
    layers: list[int] = field(default_factory=lambda: [10, 20])
    linear_name: str = "self_attn.q_proj"
    num_rotations: int = 8
    num_steps: int = 200


def main() -> None:
    config = parse(Config)
    rows = select(load_results("rotation"), linear_name=config.linear_name, alpha_mode="joint", num_rotations=config.num_rotations, num_steps=config.num_steps)

    cases = [MODEL_NAMES[model_name] for model_name in config.model_names] + ["평균"]
    fig, axes = plt.subplots(1, len(config.layers), figsize=get_figsize([get_panel_width(len(cases), len(SERIES))] * len(config.layers), 1), layout="constrained")
    for ax, layer in zip(axes, config.layers):
        # 모델마다 (schedule, arm)의 loss, random은 seed 평균, 마지막 case는 모델 평균
        losses = {series: [] for series in SERIES}
        for model_name in config.model_names:
            for schedule, arm in SERIES:
                losses[schedule, arm].append(compute_means(select(rows, model_name=model_name, layer=layer, schedule=schedule, arm=arm), "loss", ("model_name",))[arm])
        bars = [[(losses[series][case_index], SCHEDULE_COLORS[series]) for series in SERIES] for case_index in range(len(config.model_names))]
        bars.append([(mean(losses[series]), SCHEDULE_COLORS[series]) for series in SERIES])
        plot_bars(ax, cases, bars, "%")
        # 막대 사이 차이가 보이도록 가장 낮은 막대 조금 아래에서 y축을 자름
        ax.set_ylim(bottom=min(min(values) for values in losses.values()) - LOSS_YMARGIN)
        ax.set_title(f"layer {layer}")
        ax.set_ylabel("loss")

    fig.suptitle(f"{LINEAR_NAMES[config.linear_name]} rotation {config.num_rotations}개 rotation마다 다시 고르기")
    fig.legend(handles=[Patch(color=SCHEDULE_COLORS[series], label=f"{series[0]} {series[1]}") for series in SERIES], loc="outside lower center", ncol=len(SERIES))
    save_figure(fig, "rescore")


if __name__ == "__main__":
    main()
