from dataclasses import dataclass, field
from itertools import groupby, product
import time

from simple_parsing import parse
import torch

from utils.dataset import load_pile
from utils.format import format_duration, format_memory
from utils.hub import get_pending_jobs, save_rows
from utils.model import capture_linears, load_model
from utils.pairing import Arm, build_random_pairs
from utils.quantization import compute_range, quantize
from utils.train import optimize

# rotation arm의 hub 경로에 이미 저장된 alpha와 schedule 설정
ALPHA_MODE = "joint"
SCHEDULE = "all"


@dataclass
class Config:
    model_name: str = "Qwen3-0.6B"
    layers: list[int] = field(default_factory=lambda: [0])
    linear_names: list[str] = field(default_factory=lambda: ["self_attn.q_proj", "mlp.gate_proj"])
    arms: list[Arm] = field(default_factory=lambda: ["scaling", "random"])
    num_rotations: list[int] = field(default_factory=lambda: [8])
    num_bits: int = 4
    group_size: int = 128
    # calibration 데이터
    num_samples: int = 16
    sequence_len: int = 1024
    # alpha와 theta 학습
    num_steps: int = 200
    learning_rate: float = 0.01
    seeds: list[int] = field(default_factory=lambda: [42, 128, 1582, 5096, 31337])


def make_rotation_path(
    model_name: str,
    layer: int,
    linear_name: str,
    num_bits: int,
    group_size: int,
    num_samples: int,
    sequence_len: int,
    num_rotations: int | None,
    num_steps: int,
    learning_rate: float,
) -> str:
    # scaling은 rotation이 없어 alpha_mode, schedule, num_rotations를 경로에서 뺌
    # 예시) "rotation/Qwen3-0.6B/layer=0,linear_name=self_attn.q_proj,num_bits=4,group_size=128,num_samples=16,sequence_len=1024,alpha_mode=joint,schedule=all,num_rotations=8,num_steps=200,learning_rate=0.01"
    rotation_settings = "" if num_rotations is None else f"alpha_mode={ALPHA_MODE},schedule={SCHEDULE},num_rotations={num_rotations},"
    return f"rotation/{model_name}/layer={layer},linear_name={linear_name},num_bits={num_bits},group_size={group_size},num_samples={num_samples},sequence_len={sequence_len},{rotation_settings}num_steps={num_steps},learning_rate={learning_rate:g}"


def make_jobs(
    config: Config,
) -> dict[tuple[str, str, int], tuple[int, str, str, int | None, int]]:
    # (hub 경로, arm, seed)마다 (layer, linear, arm, rotation 수, seed)
    # scaling은 rotation 수가 없고, seed는 random에만 닿으므로 다른 arm은 첫 seed로 한 번만
    jobs = {}
    for layer, linear_name, arm in product(config.layers, config.linear_names, config.arms):
        num_rotations_list = [None] if arm == "scaling" else config.num_rotations
        seeds = config.seeds if arm == "random" else config.seeds[:1]
        for num_rotations, seed in product(num_rotations_list, seeds):
            hub_path = make_rotation_path(config.model_name, layer, linear_name, config.num_bits, config.group_size, config.num_samples, config.sequence_len, num_rotations, config.num_steps, config.learning_rate)
            jobs[hub_path, arm, seed] = (layer, linear_name, arm, num_rotations, seed)
    return jobs


def get_hub_path(
    job: tuple[str, str, int],
) -> str:
    return job[0]


def main() -> None:
    config = parse(Config)
    jobs = make_jobs(config)
    pending_jobs = get_pending_jobs(list(jobs))
    print(f"pending {len(pending_jobs)}/{len(jobs)}")
    if not pending_jobs:
        return

    model, tokenizer = load_model(config.model_name)
    captured = capture_linears(model, load_pile(tokenizer, config.num_samples, config.sequence_len), config.layers, config.linear_names)
    del model

    for (layer, linear_name), (x, weight) in captured.items():
        # 1. 변환 없이 양자화한 rtn을 loss와 range의 기준으로 씀
        loss_rtn = (x @ (quantize(weight, config.group_size, config.num_bits) - weight).T).pow(2).mean().item()
        range_rtn = compute_range(weight, config.group_size)

        # 2. 설정 하나의 arm과 seed를 모두 돌린 뒤 파일 하나로 올림
        linear_jobs = sorted(job for job in pending_jobs if jobs[job][:2] == (layer, linear_name))
        for hub_path, path_jobs in groupby(linear_jobs, key=get_hub_path):
            rows = []
            for job in path_jobs:
                _, _, arm, num_rotations, seed = jobs[job]
                start_time = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                pairs, is_dummy = None, None
                if arm == "random":
                    pairs, is_dummy = build_random_pairs(weight.shape[1], config.group_size, num_rotations, seed)
                    pairs, is_dummy = pairs.to(weight.device), is_dummy.to(weight.device)
                fit = optimize(weight, x, pairs, is_dummy, config.group_size, config.num_bits, config.num_steps, config.learning_rate)
                # range는 양자화기가 실제로 본 weight로 잼
                duration = time.perf_counter() - start_time
                result = {
                    "loss": fit.loss / loss_rtn,
                    "range": compute_range(fit.transformed, config.group_size) / range_rtn,
                    "num_dummies": None if is_dummy is None else int(is_dummy.sum()),
                    "total_steps": config.num_steps,
                    "duration": duration,
                }
                print(f"{hub_path} arm={arm} seed={seed} loss {result['loss']:.1%} range {result['range']:.1%} time {format_duration(duration)} memory {format_memory(torch.cuda.max_memory_allocated())}")
                rows.append({"arm": arm, "seed": seed, **result})
            save_rows(hub_path, rows)


if __name__ == "__main__":
    main()
