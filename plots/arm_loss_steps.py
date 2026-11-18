from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.plot import ARM_COLORS, LINEAR_NAMES, MODEL_NAMES, add_arm_legend, compute_means, get_figsize, get_panel_width, plot_bars, save_figure, select


@dataclass
class Config:
    model_names: list[str] = field(default_factory=lambda: ["Llama-3.1-8B", "Mistral-7B-v0.3"])
    layer: int = 10
    linear_name: str = "self_attn.q_proj"
    # magnitude_similarity는 dummy가 많아 rotation을 덜 한 탓과 구분되지 않으므로 뺌
    arms: list[str] = field(default_factory=lambda: ["butterfly", "magnitude_difference", "correlation", "activation_magnitude_difference"])
    num_rotations: int = 8
    num_steps: list[int] = field(default_factory=lambda: [200, 800])


def main() -> None:
    config = parse(Config)
    rows = select(load_results("rotation"), layer=config.layer, linear_name=config.linear_name, alpha_mode="joint", schedule="all", num_rotations=config.num_rotations)

    cases = [f"step {num_steps}" for num_steps in config.num_steps]
    fig, axes = plt.subplots(1, len(config.model_names), figsize=get_figsize([get_panel_width(len(cases), len(config.arms))] * len(config.model_names), 1), layout="constrained")
    for ax, model_name in zip(axes, config.model_names):
        # step 수마다 random 대비 loss, 0보다 크면 random보다 나쁨
        bars = []
        for num_steps in config.num_steps:
            means = compute_means(select(rows, model_name=model_name, num_steps=num_steps), "loss", ("model_name",))
            bars.append([(means[arm] - means["random"], ARM_COLORS[arm]) for arm in config.arms])
        plot_bars(ax, cases, bars, "%p")
        ax.set_title(MODEL_NAMES[model_name])
        ax.set_ylabel("random 대비 loss")

    fig.suptitle(f"{LINEAR_NAMES[config.linear_name]} layer {config.layer} rotation {config.num_rotations}개 arm별 loss")
    add_arm_legend(fig, config.arms)
    save_figure(fig, "arm_loss_steps")


if __name__ == "__main__":
    main()
