from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import AdamW

from utils.givens import rotate
from utils.quantization import quantize


@dataclass
class Fit:
    loss: float
    # loss가 가장 낮았던 step의 양자화 직전 weight
    transformed: Tensor


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
