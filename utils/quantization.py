import torch
from torch import Tensor
from torch.autograd import Function

# range가 0인 group도 0으로 나누지 않도록 range 최솟값
MIN_SCALE_RANGE = 1e-5


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
