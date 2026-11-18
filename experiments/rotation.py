from dataclasses import dataclass, field
from itertools import groupby, product
import random
import time
from typing import Literal

from datasets import load_dataset
from simple_parsing import parse
import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from utils.format import format_duration, format_memory
from utils.hub import get_pending_jobs, save_rows

MODEL_ORGS = {"Qwen3-0.6B": "Qwen"}
# scaling은 rotation 없이 alpha만 학습, random은 pair를 무작위로 고르는 현행 구현
Arm = Literal["scaling", "random"]
# rotation arm의 hub 경로에 이미 저장된 alpha와 schedule 설정
ALPHA_MODE = "joint"
SCHEDULE = "all"
# range가 0인 group도 0으로 나누지 않도록 range 최솟값
MIN_SCALE_RANGE = 1e-5


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


@dataclass
class Fit:
    loss: float
    # loss가 가장 낮았던 step의 양자화 직전 weight
    transformed: Tensor


def load_model(
    model_name: str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_ORGS[model_name]}/{model_name}")
    model = AutoModelForCausalLM.from_pretrained(f"{MODEL_ORGS[model_name]}/{model_name}", dtype=torch.bfloat16, device_map="auto")
    model.requires_grad_(False)
    return model, tokenizer


def load_pile(
    tokenizer: PreTrainedTokenizerBase,
    num_samples: int,
    sequence_len: int,
) -> Tensor:
    # paroquant appendix a.2를 따라 pile 문서를 이어 붙여 sequence_len씩 자름
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    num_tokens = num_samples * sequence_len
    token_ids = []
    for row in dataset:
        token_ids += tokenizer.encode(row["text"] + "\n\n", add_special_tokens=False)
        if len(token_ids) >= num_tokens:
            break
    return torch.tensor(token_ids[:num_tokens]).view(num_samples, sequence_len)


def capture_linears(
    model: PreTrainedModel,
    samples: Tensor,
    layers: list[int],
    linear_names: list[str],
) -> dict[tuple[int, str], tuple[Tensor, Tensor]]:
    # (layer, linear)마다 (입력 activation, weight), 이후에는 모델이 필요 없음
    linears = {model.get_submodule(f"model.layers.{layer}.{linear_name}"): (layer, linear_name) for layer in layers for linear_name in linear_names}
    inputs = {target: [] for target in linears.values()}

    def hook(
        module: Module,
        args: tuple,
    ) -> None:
        # 모델은 bf16이지만 통계와 학습은 fp32
        inputs[linears[module]].append(args[0].reshape(-1, args[0].shape[-1]).float())

    handles = [linear.register_forward_pre_hook(hook) for linear in linears]
    for input_ids in samples:
        model(input_ids[None, :].to(model.device), use_cache=False)
    for handle in handles:
        handle.remove()
    return {target: (torch.cat(inputs[target]), linear.weight.float()) for linear, target in linears.items()}


class RoundSTE(Function):
    # round는 gradient가 거의 어디서나 0이라 gradient를 그대로 흘림
    # scale과 zero에도 gradient가 가야 alpha와 theta가 range를 줄이도록 학습됨
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
    ) -> Tensor:
        return torch.round(x)

    @staticmethod
    def backward(
        ctx,
        grad: Tensor,
    ) -> Tensor:
        return grad


def quantize(
    weight: Tensor,
    group_size: int,
    num_bits: int,
) -> Tensor:
    # group마다 asymmetric min-max로 양자화한 뒤 되돌림
    groups = weight.reshape(weight.shape[0], -1, group_size)
    weight_min = groups.amin(dim=-1, keepdim=True)
    weight_max = groups.amax(dim=-1, keepdim=True)
    scale = (weight_max - weight_min).clamp(min=MIN_SCALE_RANGE) / (2**num_bits - 1)
    zero = -RoundSTE.apply(weight_min / scale)
    level = torch.clamp(RoundSTE.apply(groups / scale) + zero, 0, 2**num_bits - 1)
    return ((level - zero) * scale).reshape(weight.shape)


def compute_range(
    weight: Tensor,
    group_size: int,
) -> float:
    # 양자화 오차가 range의 제곱을 따라가므로 group별 range 제곱의 평균
    groups = weight.reshape(weight.shape[0], -1, group_size)
    return (groups.amax(dim=2) - groups.amin(dim=2)).pow(2).mean().item()


def rotate(
    weight: Tensor,
    pairs: Tensor,
    theta: Tensor,
    inverse: bool,
) -> Tensor:
    # rotation마다 pair의 두 channel을 theta만큼 돌림, inverse면 theta 부호와 rotation 순서를 함께 뒤집음
    order = range(len(pairs) - 1, -1, -1) if inverse else range(len(pairs))
    for rotation_index in order:
        first = pairs[rotation_index, :, 0]
        second = pairs[rotation_index, :, 1]
        cos = theta[rotation_index].cos()
        sin = -theta[rotation_index].sin() if inverse else theta[rotation_index].sin()
        first_columns = weight[:, first]
        second_columns = weight[:, second]
        # first를 제자리에서 바꾸면 second 계산에 섞이므로 사본에 씀
        weight = weight.clone()
        weight[:, first] = cos * first_columns - sin * second_columns
        weight[:, second] = sin * first_columns + cos * second_columns
    return weight


def build_random_pairs(
    num_columns: int,
    group_size: int,
    num_rotations: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    # group마다 모든 pair를 섞고 앞에서부터 고름
    # 한 rotation 안에서는 channel을 한 번씩만, rotation끼리는 같은 pair를 다시 쓰지 않음
    # 짝을 못 찾은 channel은 자리만 채우고 rotation하지 않는 dummy
    # 예시) group_size 4 → rotation마다 pair 2개, 예) [(0, 2), (1, 3)]
    generator = random.Random(seed)
    pairs = torch.empty(num_rotations, num_columns // 2, 2, dtype=torch.long)
    is_dummy = torch.zeros(num_rotations, num_columns // 2, dtype=torch.bool)
    for channel_offset in range(0, num_columns, group_size):
        pair_offset = channel_offset // 2
        first, second = torch.triu_indices(group_size, group_size, offset=1)
        candidates = list(zip(first.tolist(), second.tolist()))
        generator.shuffle(candidates)
        used_pairs = set()
        for rotation_index in range(num_rotations):
            used_channels = set()
            picks = []
            for candidate in candidates:
                if len(picks) * 2 == group_size:
                    break
                if candidate in used_pairs or candidate[0] in used_channels or candidate[1] in used_channels:
                    continue
                picks.append(candidate)
                used_channels.update(candidate)
                used_pairs.add(candidate)
            leftovers = [channel for channel in range(group_size) if channel not in used_channels]
            num_picks = len(picks)
            picks += [(leftovers[index], leftovers[index + 1]) for index in range(0, len(leftovers), 2)]
            pairs[rotation_index, pair_offset : pair_offset + len(picks)] = channel_offset + torch.tensor(picks)
            is_dummy[rotation_index, pair_offset + num_picks : pair_offset + len(picks)] = True
    return pairs, is_dummy


def optimize(
    weight: Tensor,
    x: Tensor,
    pairs: Tensor | None,
    is_dummy: Tensor | None,
    group_size: int,
    num_bits: int,
    num_steps: int,
    learning_rate: float,
) -> Fit:
    # paroquant appendix a.2처럼 양자화가 linear 출력에 낸 차이를 adamw로 줄임
    # pairs가 없으면 rotation 없이 alpha만 학습
    alpha = torch.ones(weight.shape[1], device=weight.device, requires_grad=True)
    parameters = [alpha]
    if pairs is not None:
        theta = torch.zeros(pairs.shape[:2], device=weight.device, requires_grad=True)
        parameters.append(theta)
    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=0)

    best = None
    for _ in range(num_steps):
        # 1. alpha, rotation, 양자화, 역rotation, 역alpha 순, dummy는 theta를 0으로 눌러 rotation에서 뺌
        transformed = weight * alpha
        if pairs is not None:
            transformed = rotate(transformed, pairs, theta * ~is_dummy, inverse=False)
        dequantized = quantize(transformed, group_size, num_bits)
        if pairs is not None:
            dequantized = rotate(dequantized, pairs, theta * ~is_dummy, inverse=True)
        dequantized = dequantized / alpha

        # 2. 변환은 되돌렸으므로 남는 것은 양자화 오차가 출력에 낸 차이뿐
        loss = (x @ (dequantized - weight).T).pow(2).mean()
        # step 뒤에 담으면 alpha와 theta만 transformed보다 한 step 앞서므로 갱신 전에 담음
        if best is None or loss.item() < best.loss:
            best = Fit(loss.item(), transformed.detach().clone())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return best


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
