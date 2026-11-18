from dataclasses import dataclass, field
from itertools import groupby, product
from typing import get_args

from simple_parsing import parse
import torch
from torch import Tensor

from utils.dataset import load_pile
from utils.hub import get_hub_path, get_pending_jobs, save_rows
from utils.model import capture_linears, load_model
from utils.pairing import Arm, build_pairs

# laplacian 고유값을 0으로 볼 상한
ZERO_EIGENVALUE = 1e-8


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    layers: list[int] = field(default_factory=lambda: [0, 10, 20])
    linear_names: list[str] = field(default_factory=lambda: ["self_attn.q_proj"])
    # scaling은 pair를 고르지 않으므로 연결성도 없음
    arms: list[Arm] = field(default_factory=lambda: [arm for arm in get_args(Arm) if arm != "scaling"])
    num_rotations: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    group_size: int = 128
    # calibration 데이터
    num_samples: int = 16
    sequence_len: int = 1024
    seeds: list[int] = field(default_factory=lambda: [42])


def compute_connectivity(
    pairs: Tensor,
    is_dummy: Tensor,
    num_columns: int,
    group_size: int,
) -> dict[str, float]:
    # 모든 rotation의 pair를 겹쳐 그래프의 edge로 보고, dummy는 뺌
    # blocks는 group당 연결된 덩어리 수, lambda2는 laplacian의 두 번째로 작은 고유값의 group 평균
    live_pairs = [tuple(pair) for pair, dummy in zip(pairs.flatten(0, 1).tolist(), is_dummy.flatten().tolist()) if not dummy]
    num_blocks = 0
    lambda2s = []
    for channel_offset in range(0, num_columns, group_size):
        # butterfly는 stride가 돌아 같은 pair가 다시 나오므로 set으로 중복을 없앰
        edges = {(first - channel_offset, second - channel_offset) for first, second in live_pairs if channel_offset <= first < channel_offset + group_size}
        adjacency = torch.zeros(group_size, group_size, dtype=torch.float64)
        for first, second in edges:
            adjacency[first, second] = adjacency[second, first] = 1
        # laplacian 고유값 0의 개수가 덩어리 수, 고유값은 오름차순이고 첫 번째는 항상 0
        eigenvalues = torch.linalg.eigvalsh(torch.diag(adjacency.sum(1)) - adjacency)
        num_blocks += int((eigenvalues < ZERO_EIGENVALUE).sum())
        lambda2s.append(eigenvalues[1].item())
    num_groups = num_columns // group_size
    return {"blocks": num_blocks / num_groups, "lambda2": sum(lambda2s) / num_groups}


def make_connectivity_path(
    model_name: str,
    layer: int,
    linear_name: str,
    group_size: int,
    num_samples: int,
    sequence_len: int,
    num_rotations: int,
) -> str:
    # 예시) "connectivity/Qwen3-0.6B/layer=10,linear_name=self_attn.q_proj,group_size=128,num_samples=16,sequence_len=1024,num_rotations=8"
    return f"connectivity/{model_name}/layer={layer},linear_name={linear_name},group_size={group_size},num_samples={num_samples},sequence_len={sequence_len},num_rotations={num_rotations}"


@torch.no_grad()
def main() -> None:
    config = parse(Config)
    # (hub 경로, arm, seed)마다 (layer, linear, arm, rotation 수, seed), seed는 random에만 닿으므로 다른 arm은 첫 seed로 한 번만
    jobs = {}
    for layer, linear_name, arm in product(config.layers, config.linear_names, config.arms):
        for num_rotations, seed in product(config.num_rotations, config.seeds if arm == "random" else config.seeds[:1]):
            hub_path = make_connectivity_path(config.model_name, layer, linear_name, config.group_size, config.num_samples, config.sequence_len, num_rotations)
            jobs[hub_path, arm, seed] = (layer, linear_name, arm, num_rotations, seed)
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
            layer, linear_name, arm, num_rotations, seed = jobs[job]
            x, weight = captured[layer, linear_name]
            pairs, is_dummy = build_pairs(weight, x.abs().mean(0), config.group_size, num_rotations, arm, seed)
            result = compute_connectivity(pairs, is_dummy, weight.shape[1], config.group_size)
            print(f"{hub_path} arm={arm} seed={seed} blocks {result['blocks']:.1f} lambda2 {result['lambda2']:.2f}")
            rows.append({"arm": arm, "seed": seed, **result})
        save_rows(hub_path, rows)


if __name__ == "__main__":
    main()
