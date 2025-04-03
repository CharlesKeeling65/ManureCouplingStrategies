"""
空间操作模块，提供距离计算策略和栅格空间操作函数
"""

import math
import numpy as np
from typing import Tuple, Generator, List, Union, Callable


class DistanceStrategy:
    """基于不同算法的距离计算策略的基类"""

    def get_directions(
        self, distance: int
    ) -> Union[Generator[Tuple[int, int], None, None], List[Tuple[int, int]]]:
        """
        计算特定距离下的方向

        Args:
            distance: 要计算方向的距离

        Returns:
            方向元组 (dr, dc) 的生成器或列表
        """
        raise NotImplementedError("子类必须实现此方法")


class UnlimitedDistanceStrategy(DistanceStrategy):
    """
    基于 new_transport_unlimited.py 中的无限制传输距离计算策略
    使用基于 sqrt(2)/2 偏移的精确距离计算
    """

    def get_directions(self, distance: int) -> Generator[Tuple[int, int], None, None]:
        """
        使用无限制传输方法计算特定距离的方向

        Args:
            distance: 要计算方向的距离

        Returns:
            表示方向的元组 (dr, dc) 的生成器
        """
        sqrt_2_div_2 = math.sqrt(2) / 2
        max_square_value = (distance + sqrt_2_div_2) ** 2
        min_square_value = (distance - 1 + sqrt_2_div_2) ** 2

        for dr in range(-distance, distance + 1):
            max_square = max_square_value - dr**2
            min_square = min_square_value - dr**2
            dc_max = int(np.sqrt(max_square))

            if min_square < 0:
                dc_min = 0
                for dc in range(-dc_max, dc_max + 1):
                    yield dr, dc
            else:
                dc_min = int(math.ceil(math.sqrt(min_square)))
                for dc in list(range(-dc_max, -dc_min + 1)) + list(
                    range(dc_min, dc_max + 1)
                ):
                    yield dr, dc


class SimpleDistanceStrategy(DistanceStrategy):
    """
    基于其他传输脚本中的距离计算策略
    使用 itertools.product 和距离过滤
    """

    def get_directions(self, distance: int) -> List[Tuple[int, int]]:
        """
        使用简单距离计算方法计算特定距离的方向

        Args:
            distance: 要计算方向的距离

        Returns:
            表示方向的元组 (dr, dc) 的列表
        """
        import itertools

        return [
            (dr, dc)
            for dr, dc in itertools.product(
                range(-distance, distance + 1), range(-distance, distance + 1)
            )
            if math.sqrt(dr**2 + dc**2) - math.sqrt(2) / 2 <= distance
            and math.sqrt(dr**2 + dc**2) - math.sqrt(2) / 2 > (distance - 1)
        ]


def move_array(
    arr: np.ndarray, direct: Tuple[int, int], max_distance: int
) -> np.ndarray:
    """
    向指定方向移动数组，并进行边缘填充

    Args:
        arr: 要移动的输入数组
        direct: 方向元组 (di, dj)
        max_distance: 用于填充的最大距离

    Returns:
        移动后并移除填充的数组
    """
    di, dj = direct
    padded_arr = np.pad(arr, max_distance, mode="constant").astype(np.float32)
    moved_arr = np.roll(padded_arr, (di, dj), axis=(0, 1)).astype(np.float32)
    return moved_arr[max_distance:-max_distance, max_distance:-max_distance]


def edges_line_func(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    创建正值和负值之间连接的二进制掩码

    Args:
        p: 正值（盈余）数组
        n: 负值（需求）数组

    Returns:
        p和n都有非零值的位置处为1，其他位置为0的二进制掩码
    """
    return np.where((p != 0) & (n != 0), 1, 0).astype(np.int8)


def edges_weight_func(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    计算正值和负值之间的边权重

    Args:
        p: 正值（盈余）数组
        n: 负值（需求）数组

    Returns:
        边权重数组
    """
    return abs(p) + abs(n) - abs(p + n)
