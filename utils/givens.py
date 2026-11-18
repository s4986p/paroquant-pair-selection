from torch import Tensor


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
