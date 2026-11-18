from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.plot import ARM_COLORS, ARM_LABELS, LINEAR_NAMES, LOSS_YMARGIN, add_arm_legend, compute_means, get_figsize, get_panel_width, plot_bars, save_figure, select


@dataclass
class Config:
    layers: list[int] = field(default_factory=lambda: [0, 10, 20])
    linear_name: str = "self_attn.q_proj"
    arms: list[str] = field(default_factory=lambda: ["random", "butterfly", "magnitude_difference", "magnitude_similarity", "correlation", "activation_magnitude_difference"])
    num_rotations: int = 8
    num_steps: int = 200


def main() -> None:
    config = parse(Config)
    rows = select(load_results("rotation"), linear_name=config.linear_name, arm=config.arms, alpha_mode="joint", schedule="all", num_rotations=config.num_rotations, num_steps=config.num_steps)
    num_models = len({row["model_name"] for row in rows})

    fig, axes = plt.subplots(1, len(config.layers), figsize=get_figsize([get_panel_width(len(config.arms), 1)] * len(config.layers), 1), layout="constrained")
    for ax, layer in zip(axes, config.layers):
        means = compute_means(select(rows, layer=layer), "loss", ("model_name",))
        plot_bars(ax, [ARM_LABELS[arm] for arm in config.arms], [[(means[arm], ARM_COLORS[arm])] for arm in config.arms], "%")
        # layer마다 loss 높이가 달라 y축을 따로 두고, arm 사이 차이가 보이도록 가장 낮은 막대 조금 아래에서 자름
        ax.set_ylim(bottom=min(means.values()) - LOSS_YMARGIN)
        ax.set_title(f"layer {layer}")
        ax.set_ylabel("loss")

    fig.suptitle(f"{LINEAR_NAMES[config.linear_name]} rotation {config.num_rotations}개, 모델 {num_models}개 평균 arm별 loss")
    add_arm_legend(fig, config.arms)
    save_figure(fig, "arm_loss")


if __name__ == "__main__":
    main()
