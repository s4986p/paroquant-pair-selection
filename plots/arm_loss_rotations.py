from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.plot import ARM_COLORS, LINEAR_NAMES, add_arm_legend, compute_means, get_figsize, get_panel_width, plot_bars, save_figure, select


@dataclass
class Config:
    layers: list[int] = field(default_factory=lambda: [10, 20])
    linear_name: str = "self_attn.q_proj"
    # magnitude_similarity는 dummy가 많아 rotation을 덜 한 탓과 구분되지 않으므로 뺌
    arms: list[str] = field(default_factory=lambda: ["butterfly", "magnitude_difference", "correlation", "activation_magnitude_difference", "oracle"])
    num_rotations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_steps: int = 200


def main() -> None:
    config = parse(Config)
    rows = select(load_results("rotation"), linear_name=config.linear_name, alpha_mode="joint", schedule="all", num_rotations=config.num_rotations, num_steps=config.num_steps)
    num_models = len({row["model_name"] for row in rows})

    cases = [f"rotation {num_rotations}개" for num_rotations in config.num_rotations]
    fig, axes = plt.subplots(1, len(config.layers), figsize=get_figsize([get_panel_width(len(cases), len(config.arms))] * len(config.layers), 1), layout="constrained")
    for ax, layer in zip(axes, config.layers):
        # rotation 수마다 random 대비 loss, 0보다 크면 random보다 나쁨
        bars = []
        for num_rotations in config.num_rotations:
            means = compute_means(select(rows, layer=layer, num_rotations=num_rotations), "loss", ("model_name",))
            bars.append([(means[arm] - means["random"], ARM_COLORS[arm]) for arm in config.arms])
        plot_bars(ax, cases, bars, "%p")
        ax.set_title(f"layer {layer}")
        ax.set_ylabel("random 대비 loss")

    fig.suptitle(f"{LINEAR_NAMES[config.linear_name]} 모델 {num_models}개 평균 arm별 loss")
    add_arm_legend(fig, config.arms)
    save_figure(fig, "arm_loss_rotations")


if __name__ == "__main__":
    main()
