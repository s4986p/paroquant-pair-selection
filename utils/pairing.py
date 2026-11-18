import random
from typing import Literal

import torch
from torch import Tensor

# scaling은 rotation 없이 alpha만 학습, random은 pair를 무작위로 고르는 현행 구현, butterfly는 통계를 보지 않는 고정 패턴
# 나머지는 pair 점수가 큰 순서로 고르는 정렬 arm, oracle은 대신 range를 직접 봄
Arm = Literal["scaling", "random", "butterfly", "magnitude_difference", "magnitude_similarity", "correlation", "activation_magnitude_difference", "oracle"]
# oracle이 pair마다 훑는 각도 수, theta는 나중에 다시 학습하므로 순위만 나오면 됨
NUM_ORACLE_ANGLES = 16
# pair를 한 번에 다 잡으면 행 × pair × 각도라 메모리 때문에 이만큼씩 나눔
ORACLE_CHUNK_SIZE = 512
# rotation은 두 channel만 바꾸므로, 나머지의 최대와 최소는 상위와 하위 3개만 있으면 남음
NUM_EXTREMES = 3


def compute_oracle_score(
    group: Tensor,
    first: Tensor,
    second: Tensor,
) -> Tensor:
    # pair를 가장 좋은 각도로 한 번 돌렸을 때 줄어드는 range 제곱의 행 평균
    angles = torch.arange(NUM_ORACLE_ANGLES, device=group.device) * (2 * torch.pi / NUM_ORACLE_ANGLES)
    cos, sin = angles.cos(), angles.sin()
    top_values, top_indices = group.topk(NUM_EXTREMES, dim=1)
    bottom_values, bottom_indices = group.topk(NUM_EXTREMES, dim=1, largest=False)
    original_range = (top_values[:, 0] - bottom_values[:, 0]).pow(2)

    scores = []
    for start in range(0, len(first), ORACLE_CHUNK_SIZE):
        chunk_first = first[start : start + ORACLE_CHUNK_SIZE]
        chunk_second = second[start : start + ORACLE_CHUNK_SIZE]
        # 1. pair가 빼간 자리를 건너뛰고 남는 최대와 최소
        keeps_top = (top_indices[:, None, :] != chunk_first[None, :, None]) & (top_indices[:, None, :] != chunk_second[None, :, None])
        keeps_bottom = (bottom_indices[:, None, :] != chunk_first[None, :, None]) & (bottom_indices[:, None, :] != chunk_second[None, :, None])
        rest_max = torch.where(keeps_top, top_values[:, None, :], -torch.inf).amax(dim=2)
        rest_min = torch.where(keeps_bottom, bottom_values[:, None, :], torch.inf).amin(dim=2)

        # 2. 각도마다 돌린 두 channel과 나머지로 range를 다시 재고, 가장 많이 줄어든 각도의 값
        first_columns = group[:, chunk_first]
        second_columns = group[:, chunk_second]
        best = None
        for angle_cos, angle_sin in zip(cos, sin):
            rotated_first = angle_cos * first_columns - angle_sin * second_columns
            rotated_second = angle_sin * first_columns + angle_cos * second_columns
            high = torch.maximum(rest_max, torch.maximum(rotated_first, rotated_second))
            low = torch.minimum(rest_min, torch.minimum(rotated_first, rotated_second))
            drop = (original_range[:, None] - (high - low).pow(2)).mean(0)
            best = drop if best is None else torch.maximum(best, drop)
        scores.append(best)
    return torch.cat(scores)


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
    if arm == "activation_magnitude_difference":
        # weight 크기에 입력 크기를 곱함
        magnitude = group.abs().mean(0) * group_input_magnitude
        return (magnitude[first] - magnitude[second]).abs()
    return compute_oracle_score(group, first, second)


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
