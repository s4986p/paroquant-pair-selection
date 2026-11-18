import random
from typing import Literal

import torch
from torch import Tensor

# scaling은 rotation 없이 alpha만 학습, random은 pair를 무작위로 고르는 현행 구현, butterfly는 통계를 보지 않는 고정 패턴
# 나머지는 pair 점수가 큰 순서로 고르는 정렬 arm
Arm = Literal["scaling", "random", "butterfly", "magnitude_difference", "magnitude_similarity", "correlation", "activation_magnitude_difference"]


def compute_pair_score(
    group: Tensor,
    group_input_magnitude: Tensor,
    arm: Arm,
    first: Tensor,
    second: Tensor,
) -> Tensor:
    # 정렬 arm이 pair를 줄 세우는 점수, 클수록 먼저 고름, first와 second는 group 안 channel index
    if arm == "magnitude_difference":
        magnitude = group.abs().mean(0)
        return (magnitude[first] - magnitude[second]).abs()
    if arm == "magnitude_similarity":
        magnitude = group.abs().mean(0)
        return -(magnitude[first] - magnitude[second]).abs()
    if arm == "correlation":
        return torch.corrcoef(group.T).nan_to_num()[first, second].abs()
    # activation_magnitude_difference는 weight 크기에 입력 크기를 곱함
    magnitude = group.abs().mean(0) * group_input_magnitude
    return (magnitude[first] - magnitude[second]).abs()


def build_pairs(
    weight: Tensor,
    input_magnitude: Tensor,
    group_size: int,
    num_rotations: int,
    arm: Arm,
    seed: int,
) -> tuple[Tensor, Tensor]:
    # rotation마다 group 안 channel을 모두 pair로 묶음, pair는 group 안에서만 지음
    # 예시) group_size 4 → rotation마다 pair 2개, 예) [(0, 2), (1, 3)]
    generator = random.Random(seed)
    num_columns = weight.shape[1]
    pairs = torch.empty(num_rotations, num_columns // 2, 2, dtype=torch.long)
    is_dummy = torch.zeros(num_rotations, num_columns // 2, dtype=torch.bool)
    # butterfly stride 수, group_size 128이면 7개라 8번째 rotation은 첫 번째와 같은 pair
    num_strides = group_size.bit_length() - 1
    for channel_offset in range(0, num_columns, group_size):
        pair_offset = channel_offset // 2

        # 1. butterfly는 channel 번호를 stride로 xor해 짝지어, group마다 같은 pair를 씀
        # 예시) stride 1 → (0, 1), (2, 3), ...
        if arm == "butterfly":
            for rotation_index in range(num_rotations):
                stride = 1 << (rotation_index % num_strides)
                picks = [(channel, channel ^ stride) for channel in range(group_size) if channel < channel ^ stride]
                pairs[rotation_index, pair_offset : pair_offset + len(picks)] = channel_offset + torch.tensor(picks)
            continue

        # 2. random은 group 안 모든 pair를 섞고, 정렬 arm은 pair 점수가 큰 순서로 줄 세움
        first, second = torch.triu_indices(group_size, group_size, offset=1, device=weight.device)
        if arm == "random":
            candidates = list(zip(first.tolist(), second.tolist()))
            generator.shuffle(candidates)
        else:
            group = weight[:, channel_offset : channel_offset + group_size]
            order = compute_pair_score(group, input_magnitude[channel_offset : channel_offset + group_size], arm, first, second).argsort(descending=True)
            candidates = list(zip(first[order].tolist(), second[order].tolist()))

        # 3. 앞에서부터 고름, 한 rotation 안에서는 channel을 한 번씩만, rotation끼리는 같은 pair를 다시 쓰지 않음
        # 짝을 못 찾은 channel은 자리만 채우고 rotation하지 않는 dummy
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
