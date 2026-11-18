from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from simple_parsing import parse

from utils.hub import load_results
from utils.plot import ARM_COLORS, ARM_LABELS, LINEAR_NAMES, MODEL_NAMES, add_arm_legend, compute_means, get_figsize, get_panel_width, plot_bars, save_figure, select


@dataclass
class Config:
    model_name: str = "Llama-3.1-8B"
    layer: int = 10
    linear_name: str = "self_attn.q_proj"
    # random과 butterfly는 점수가 없고, magnitude_similarity는 dummy가 많아 rotation을 덜 한 탓과 구분되지 않으므로 뺌
    arms: list[str] = field(default_factory=lambda: ["magnitude_difference", "correlation", "activation_magnitude_difference", "oracle"])
    num_rotations: int = 8
    num_steps: int = 200


def main() -> None:
    config = parse(Config)
    conditions = {"model_name": config.model_name, "layer": config.layer, "linear_name": config.linear_name, "num_rotations": config.num_rotations, "num_steps": config.num_steps}
    score_rows = select(load_results("pair_score"), **conditions)
    loss_rows = select(load_results("rotation"), alpha_mode="joint", schedule="all", **conditions)

    # 1. rotation마다 score_retention, 정렬 arm은 seed가 하나라 행 하나
    retention_cases = [f"{rotation_index + 1}번째" for rotation_index in range(config.num_rotations)]
    retention_bars = []
    for rotation_index in range(config.num_rotations):
        retention_bars.append([(select(score_rows, arm=arm)[0]["rotations"][rotation_index]["score_retention"], ARM_COLORS[arm]) for arm in config.arms])
    # 2. 같은 설정의 rotation 실험 loss, random과 비교
    loss_arms = ["random", *config.arms]
    losses = compute_means(loss_rows, "loss", ("model_name",))

    panel_widths = [get_panel_width(len(retention_cases), len(config.arms)), get_panel_width(len(loss_arms), 1)]
    fig, (retention_ax, loss_ax) = plt.subplots(1, 2, figsize=get_figsize(panel_widths, 1), width_ratios=panel_widths, layout="constrained")
    plot_bars(retention_ax, retention_cases, retention_bars, "%")
    retention_ax.set_ylabel("score_retention")
    retention_ax.set_title("rotation")
    plot_bars(loss_ax, [ARM_LABELS[arm] for arm in loss_arms], [[(losses[arm], ARM_COLORS[arm])] for arm in loss_arms], "%")
    loss_ax.set_ylim(bottom=0)
    loss_ax.set_ylabel("loss")
    loss_ax.set_title(f"rotation {config.num_rotations}개")

    fig.suptitle(f"{MODEL_NAMES[config.model_name]} {LINEAR_NAMES[config.linear_name]} layer {config.layer} pair 점수 유지")
    add_arm_legend(fig, loss_arms)
    save_figure(fig, "pair_score")


if __name__ == "__main__":
    main()
