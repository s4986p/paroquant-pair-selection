from dataclasses import dataclass, field
from statistics import mean

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.plot import ARM_COLORS, LINEAR_NAMES, add_arm_legend, get_figsize, get_panel_width, plot_bars, save_figure, select

# 연결성 지표마다 unit, blocks는 group당 덩어리 수
METRIC_UNITS = {"blocks": "개", "lambda2": ""}


@dataclass
class Config:
    layers: list[int] = field(default_factory=lambda: [10, 20])
    linear_name: str = "self_attn.q_proj"
    # magnitude_similarity는 dummy가 많아 rotation을 덜 한 탓과 구분되지 않으므로 뺌
    arms: list[str] = field(default_factory=lambda: ["random", "butterfly", "magnitude_difference", "correlation", "activation_magnitude_difference"])
    num_rotations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])


def main() -> None:
    config = parse(Config)
    rows = select(load_results("connectivity"), layer=config.layers, linear_name=config.linear_name)
    num_models = len({row["model_name"] for row in rows})

    cases = [f"rotation {num_rotations}개" for num_rotations in config.num_rotations]
    fig, axes = plt.subplots(1, len(METRIC_UNITS), figsize=get_figsize([get_panel_width(len(cases), len(config.arms))] * len(METRIC_UNITS), 1), layout="constrained")
    for ax, (metric, unit) in zip(axes, METRIC_UNITS.items()):
        # seed는 하나라 rotation 수와 arm마다 layer와 모델 평균
        bars = [[(mean(row[metric] for row in select(rows, arm=arm, num_rotations=num_rotations)), ARM_COLORS[arm]) for arm in config.arms] for num_rotations in config.num_rotations]
        plot_bars(ax, cases, bars, unit)
        ax.set_ylabel(metric)

    layers = ", ".join(str(layer) for layer in config.layers)
    fig.suptitle(f"{LINEAR_NAMES[config.linear_name]} layer {layers}, 모델 {num_models}개 평균 연결성")
    add_arm_legend(fig, config.arms)
    save_figure(fig, "connectivity")


if __name__ == "__main__":
    main()
