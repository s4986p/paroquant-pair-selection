import random
from typing import Literal

import torch
from torch import Tensor

# scaling은 rotation 없이 alpha만 학습, random은 pair를 무작위로 고르는 현행 구현
Arm = Literal["scaling", "random"]


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
