from dataclasses import dataclass, field
from itertools import groupby, product
import time
from typing import get_args

from simple_parsing import parse
import torch
from torch import Tensor

from utils.dataset import load_pile
from utils.format import format_duration, format_memory
from utils.givens import rotate
from utils.hub import get_hub_path, get_pending_jobs, save_rows
from utils.model import capture_linears, load_model
from utils.pairing import Arm, build_pairs, compute_pair_score
from utils.quantization import compute_range
from utils.train import optimize

# 점수 없이 pair를 고르는 arm, score_retention이 없음
UNSCORED_ARMS = ["random", "butterfly"]


@dataclass
class Config:
    model_name: str = "Llama-3.1-8B"
    layers: list[int] = field(default_factory=lambda: [10])
    linear_names: list[str] = field(default_factory=lambda: ["self_attn.q_proj"])
    # scaling은 pair를 고르지 않으므로 볼 것이 없음
    arms: list[Arm] = field(default_factory=lambda: [arm for arm in get_args(Arm) if arm != "scaling"])
    num_rotations: list[int] = field(default_factory=lambda: [8])
    num_bits: int = 4
    group_size: int = 128
    # calibration 데이터
    num_samples: int = 16
    sequence_len: int = 1024
    # rotation 실험과 같은 학습을 여기서도 돌려 같은 theta로 봄
    num_steps: list[int] = field(default_factory=lambda: [200])
    learning_rate: float = 0.01
    seeds: list[int] = field(default_factory=lambda: [42, 128, 1582, 5096, 31337])


def score_rotation(
    weight: Tensor,
    input_magnitude: Tensor,
    pairs: Tensor,
    is_dummy: Tensor,
    arm: Arm,
    group_size: int,
) -> Tensor:
    # rotation 하나가 고른 pair의 arm 점수를 group마다 평균
    # group마다 크기 규모가 달라 하나로 뭉개지 않고, 비율은 호출하는 쪽이 group마다 냄
    # pair는 group 안에서만 지어지므로 first의 group이 곧 pair의 group
    group_pairs = {}
    for (first, second), dummy in zip(pairs.tolist(), is_dummy.tolist()):
        if not dummy:
            group_pairs.setdefault(first // group_size, []).append((first, second))
    scores = []
    for group_index in sorted(group_pairs):
        channel_offset = group_index * group_size
        picks = group_pairs[group_index]
        first = torch.tensor([first - channel_offset for first, _ in picks], device=weight.device)
        second = torch.tensor([second - channel_offset for _, second in picks], device=weight.device)
        scores.append(compute_pair_score(weight[:, channel_offset : channel_offset + group_size], input_magnitude[channel_offset : channel_offset + group_size], arm, first, second).mean())
    return torch.stack(scores) if scores else weight.new_zeros(0)


def make_pair_score_path(
    model_name: str,
    layer: int,
    linear_name: str,
    num_bits: int,
    group_size: int,
    num_samples: int,
    sequence_len: int,
    num_rotations: int,
    num_steps: int,
    learning_rate: float,
) -> str:
    # 예시) "pair_score/Llama-3.1-8B/layer=10,linear_name=self_attn.q_proj,num_bits=4,group_size=128,num_samples=16,sequence_len=1024,num_rotations=8,num_steps=200,learning_rate=0.01"
    return f"pair_score/{model_name}/layer={layer},linear_name={linear_name},num_bits={num_bits},group_size={group_size},num_samples={num_samples},sequence_len={sequence_len},num_rotations={num_rotations},num_steps={num_steps},learning_rate={learning_rate:g}"


def main() -> None:
    config = parse(Config)
    # (hub 경로, arm, seed)마다 (layer, linear, arm, rotation 수, step 수, seed), seed는 random에만 닿으므로 다른 arm은 첫 seed로 한 번만
    jobs = {}
    for layer, linear_name, arm in product(config.layers, config.linear_names, config.arms):
        for num_rotations, num_steps, seed in product(config.num_rotations, config.num_steps, config.seeds if arm == "random" else config.seeds[:1]):
            hub_path = make_pair_score_path(config.model_name, layer, linear_name, config.num_bits, config.group_size, config.num_samples, config.sequence_len, num_rotations, num_steps, config.learning_rate)
            jobs[hub_path, arm, seed] = (layer, linear_name, arm, num_rotations, num_steps, seed)
    pending_jobs = get_pending_jobs(list(jobs))
    print(f"pending {len(pending_jobs)}/{len(jobs)}")
    if not pending_jobs:
        return

    model, tokenizer = load_model(config.model_name)
    captured = capture_linears(model, load_pile(tokenizer, config.num_samples, config.sequence_len), config.layers, config.linear_names)
    del model

    # 설정 하나의 arm과 seed를 모두 돌린 뒤 파일 하나로 올림
    for hub_path, path_jobs in groupby(sorted(pending_jobs), key=get_hub_path):
        rows = []
        for job in path_jobs:
            layer, linear_name, arm, num_rotations, num_steps, seed = jobs[job]
            start_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            x, weight = captured[layer, linear_name]
            input_magnitude = x.abs().mean(0)
            range_rtn = compute_range(weight, config.group_size)

            # 1. rotation 실험과 같이 pair를 고르고 alpha와 theta를 학습
            pairs, is_dummy = build_pairs(weight, input_magnitude, config.group_size, num_rotations, arm, seed)
            pairs, is_dummy = pairs.to(weight.device), is_dummy.to(weight.device)
            fit = optimize(weight, x, pairs, is_dummy, config.group_size, config.num_bits, num_steps, config.learning_rate)
            theta = fit.theta * ~is_dummy

            # 2. rotation을 하나씩 걸며, 그 rotation의 pair 점수를 rotation 직전 weight와 rotation 전 weight로 잼
            # alpha가 먼저 걸리고 rotation이 그 위에 걸리므로 alpha만 건 weight가 rotation 전 기준
            # score_retention은 rotation 직전 점수와 rotation 전 점수의 비로, 앞선 rotation 뒤에 점수가 얼마나 남았는지 봄
            base = weight * fit.alpha
            current = base
            rotations = []
            for rotation_index in range(num_rotations):
                row = {"num_dummies": int(is_dummy[rotation_index].sum()), "original_score": None, "score_retention": None}
                if arm not in UNSCORED_ARMS:
                    # original_score는 pair를 고를 때 본 원본 weight로 잰 점수
                    row["original_score"] = score_rotation(weight, input_magnitude, pairs[rotation_index], is_dummy[rotation_index], arm, config.group_size).mean().item()
                    base_score = score_rotation(base, input_magnitude, pairs[rotation_index], is_dummy[rotation_index], arm, config.group_size)
                    current_score = score_rotation(current, input_magnitude, pairs[rotation_index], is_dummy[rotation_index], arm, config.group_size)
                    # correlation은 기준 점수가 0일 수 있어 그런 group은 뺌
                    nonzero = base_score != 0
                    if nonzero.any():
                        row["score_retention"] = (current_score[nonzero] / base_score[nonzero]).mean().item()
                current = rotate(current, pairs[rotation_index : rotation_index + 1], theta[rotation_index : rotation_index + 1], inverse=False)
                row["range"] = compute_range(current, config.group_size) / range_rtn
                rotations.append(row)

            result = {"initial_range": compute_range(base, config.group_size) / range_rtn, "rotations": rotations}
            score_retention = "-" if rotations[-1]["score_retention"] is None else f"{rotations[-1]['score_retention']:.1%}"
            print(f"{hub_path} arm={arm} seed={seed} range {rotations[-1]['range']:.1%} score_retention {score_retention} time {format_duration(time.perf_counter() - start_time)} memory {format_memory(torch.cuda.max_memory_allocated())}")
            rows.append({"arm": arm, "seed": seed, **result})
        save_rows(hub_path, rows)


if __name__ == "__main__":
    main()
