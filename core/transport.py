"""
运输模型模块，提供各种粪肥养分运输策略的实现
"""

import profile
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from typing import List, Tuple, Dict, Optional, Union, Any
import sys
import rasterio as rio
import cvxpy as cp


def optimal_allocation_linprog(sd_raster, distance):
    """
    主分配函数，输入余缺栅格和最大运输距离，输出分配后余缺栅格。
    Main allocation function, input: supply-demand raster and max transport distance, output: allocated raster.
    使用cvxpy实现线性规划。
    """
    p_raster = np.where(sd_raster > 0, sd_raster, 0)
    n_raster = np.where(sd_raster < 0, sd_raster, 0)
    X, Y = sd_raster.shape
    rows = np.array([], dtype=np.int64)
    cols = np.array([], dtype=np.int64)
    for direct in direct_list(distance):
        weight_arr = edges_line_func(move_array(p_raster, direct, distance), n_raster)
        weight_arr_nonzero_index = np.nonzero(weight_arr)
        tar_index = weight_arr_nonzero_index[0] * Y + weight_arr_nonzero_index[1]
        sou_index = (weight_arr_nonzero_index[0] - direct[0]) * Y + (
            weight_arr_nonzero_index[1] - direct[1]
        )
        rows = np.concatenate((rows, tar_index, sou_index))
        cols = np.concatenate(
            (
                cols,
                np.arange(
                    int(len(cols) / 2),
                    len(tar_index) + int(len(cols) / 2),
                    dtype=np.int64,
                ),
                np.arange(
                    int(len(cols) / 2),
                    len(tar_index) + int(len(cols) / 2),
                    dtype=np.int64,
                ),
            )
        )
    try:
        A = csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            shape=(X * Y, int(cols[-1]) + 1),
            dtype=np.int8,
        )
        w = sd_raster.reshape(-1)
        c = -np.ones((A.shape[1],))
        x = cp.Variable(A.shape[1], nonneg=True)
        constraints = [A @ x <= np.abs(w)]
        objective = cp.Minimize(c @ x)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.HIGHS)
        delta_w = np.sign(w) * (A @ x.value)
        new_w = w - delta_w
        new_sd_raster = new_w.reshape((X, Y))
    except Exception as e:
        logging.error(f"Error in linprog: {e}, edge number: {len(rows)}")
        new_sd_raster = sd_raster
    return new_sd_raster

def direct_list(distance):
    """
    生成给定距离下的方向列表。
    Generate direction list for a given distance.
    """
    sqrt_2_div_2 = np.sqrt(2) / 2
    max_square_value = (distance + sqrt_2_div_2) ** 2
    min_square_value = (distance - 1 + sqrt_2_div_2) ** 2
    for dr in range(-distance, distance + 1):
        max_square = max_square_value - dr**2
        min_square = min_square_value - dr**2
        dc_max = int(np.sqrt(max_square))
        if min_square < 0:
            for dc in range(-dc_max, dc_max + 1):
                yield dr, dc
        else:
            dc_min = int(np.ceil(np.sqrt(min_square)))
            for dc in list(range(-dc_max, -dc_min + 1)) + list(range(dc_min, dc_max + 1)):
                yield dr, dc

def move_array(arr, direct, max_distance):
    """
    按方向移动数组。
    Move array by direction.
    """
    di, dj = direct
    padded_arr = np.pad(arr, max_distance, mode="constant").astype(np.float32)
    moved_arr = np.roll(padded_arr, (di, dj), axis=(0, 1)).astype(np.float32)
    return moved_arr[max_distance:-max_distance, max_distance:-max_distance]

def edges_line_func(p, n):
    return np.where((p != 0) & (n != 0), 1, 0).astype(np.int8)
