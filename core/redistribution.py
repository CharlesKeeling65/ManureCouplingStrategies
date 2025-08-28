"""
redistribution.py

畜禽再分配优化算法模块
Livestock redistribution optimization algorithms module
"""
import numpy as np
import itertools
import cvxpy as cp
from scipy.sparse import csr_matrix

def get_indices(raster_data, value):
    """
    获取满足条件的索引
    """
    indices = np.argwhere(raster_data == value)
    return indices

def update_inPs(
    intensive_supply_1km_animal_matrix: np.ndarray,
    adm_shp,
    adm_code_list,
    exPs,
    Pdemand_1km,
    intensive_supply_limitation_matrix,
):
    """
    全局/分区再分配主函数。
    Main function for global/region-wise livestock redistribution.
    """
    inPs = intensive_supply_1km_animal_matrix.sum(axis=0)
    s_d_raster = inPs + exPs - Pdemand_1km
    # 生成行政区掩膜（假设adm_shp.geometry为多边形列表，adm_code_list为区划码列表）
    from utils.geometry import polygon_to_mask
    adm_mask = polygon_to_mask(adm_shp.geometry, s_d_raster.shape)
    intensive_supply_max_species_order = intensive_supply_1km_animal_matrix.argmax(axis=0)
    need2out_flag = (
        (
            intensive_supply_1km_animal_matrix.max(axis=0)
            >= intensive_supply_limitation_matrix[intensive_supply_max_species_order]
        )
        * (s_d_raster > 0)
        * (adm_mask >= 0)
    )
    need2in_flag = (
        (
            (intensive_supply_1km_animal_matrix.max(axis=0) == 0)
            * (
                s_d_raster
                < -intensive_supply_limitation_matrix[intensive_supply_max_species_order]
            )
        )
        + (
            (
                intensive_supply_1km_animal_matrix.max(axis=0)
                >= intensive_supply_limitation_matrix[intensive_supply_max_species_order]
            )
            * (s_d_raster < 0)
        )
    ) * (adm_mask >= 0)
    need2out_flag = np.where(need2out_flag > 0, 1, 0)
    need2in_flag = np.where(need2in_flag > 0, 1, 0)
    need2out_indices = get_indices(need2out_flag, 1)
    need2in_indices = get_indices(need2in_flag, 1)
    need2out_speices_order = intensive_supply_max_species_order[tuple(need2out_indices.T)]
    need2in_speices_order = intensive_supply_max_species_order[tuple(need2in_indices.T)]
    need2out_adm_code = adm_mask[tuple(need2out_indices.T)]
    need2in_adm_code = adm_mask[tuple(need2in_indices.T)]
    need2out_inS = intensive_supply_1km_animal_matrix[
        tuple(
            np.array(
                [
                    [order, indice[0], indice[1]]
                    for order, indice in zip(need2out_speices_order, need2out_indices)
                ]
            ).T
        )
    ]
    need2out_limit_S = intensive_supply_limitation_matrix[need2out_speices_order]
    need2out_s = s_d_raster[tuple(need2out_indices.T)]
    need2in_inS = intensive_supply_1km_animal_matrix[
        tuple(
            np.array(
                [
                    [order, indice[0], indice[1]]
                    for order, indice in zip(need2in_speices_order, need2in_indices)
                ]
            ).T
        )
    ]
    need2in_limit_S = intensive_supply_limitation_matrix[need2in_speices_order]
    need2in_s = s_d_raster[tuple(need2in_indices.T)]
    # 下界和上界
    need2out_delta_lb = np.maximum(
        -(
            np.where(
                need2out_inS - need2out_limit_S < 0, 0, need2out_inS - need2out_limit_S
            )
        ),
        -need2out_s,
    )
    need2out_delta_ub = np.zeros_like(need2out_delta_lb)
    need2in_delta_lb = np.where(
        need2in_limit_S - need2in_inS < 0,
        0,
        need2in_limit_S - need2in_inS,
    )
    need2in_delta_ub = -need2in_s

    need2in_delta_is_lb_ub_zeros_index = np.array([], dtype=np.int32)
    sd_list = [
        (
            np.where(
                (need2out_adm_code == i) & (need2out_speices_order == j),
                need2out_delta_lb,
                0,
            ),
            np.where(
                (need2in_adm_code == i) & (need2in_speices_order == j),
                need2in_delta_lb,
                0,
            ),
        )
        for i, j in itertools.product(range(len(adm_code_list)), range(intensive_supply_1km_animal_matrix.shape[0]))
        if -np.where(
            (need2out_adm_code == i) & (need2out_speices_order == j),
            need2out_delta_lb,
            0,
        ).sum()
        < np.where(
            (need2in_adm_code == i) & (need2in_speices_order == j), need2in_delta_lb, 0
        ).sum()
    ]
    for s, d in sd_list:
        sorted_need2in_lb = np.sort(d)[::-1]
        sorted_need2in_lb_index = np.argsort(d)[::-1]
        sorted_need2in_lb_cumsum = np.cumsum(sorted_need2in_lb)
        zeros_index = np.setdiff1d(
            sorted_need2in_lb_index[
                np.argwhere((-s.sum() - sorted_need2in_lb_cumsum) < 0).reshape(-1)
            ],
            sorted_need2in_lb_index[np.argwhere(sorted_need2in_lb == 0).reshape(-1)],
        )
        need2in_delta_is_lb_ub_zeros_index = np.concatenate(
            [need2in_delta_is_lb_ub_zeros_index, zeros_index]
        )
    need2in_delta_is_lb_ub_zeros_index = np.unique(need2in_delta_is_lb_ub_zeros_index)
    need2in_delta_lb[need2in_delta_is_lb_ub_zeros_index] = 0
    need2in_delta_ub[need2in_delta_is_lb_ub_zeros_index] = 0
    all_delta_lb = np.concatenate([need2out_delta_lb, need2in_delta_lb], axis=0)
    all_delta_ub = np.concatenate([need2out_delta_ub, need2in_delta_ub], axis=0)
    # 构建A矩阵
    all_indices = np.concatenate([need2out_indices, need2in_indices], axis=0)
    all_adm_code = np.concatenate([need2out_adm_code, need2in_adm_code], axis=0)
    all_speices_order = np.concatenate([need2out_speices_order, need2in_speices_order], axis=0)
    A_rows = all_adm_code * intensive_supply_1km_animal_matrix.shape[0] + all_speices_order
    A_cols = np.array([i for i in range(len(all_indices))], dtype=np.int32)
    A = csr_matrix(
        (np.ones_like(A_rows), (A_rows, A_cols)),
        shape=(len(adm_code_list) * intensive_supply_1km_animal_matrix.shape[0], len(all_indices)),
        dtype=np.int8,
    )
    n = len(all_indices)
    x = cp.Variable(n)
    all_s = s_d_raster[tuple(all_indices.T)]
    objective = cp.Minimize(cp.norm1(all_s + x))
    constraints = [
        A @ x == 0,
        x >= all_delta_lb,
        x <= all_delta_ub,
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.HIGHS, verbose=False)
    # 更新供给矩阵
    intensive_supply_1km_animal_matrix[
        tuple(
            np.array(
                [
                    [order, indice[0], indice[1]]
                    for order, indice in zip(all_speices_order, all_indices)
                ]
            ).T
        )
    ] += x.value
    return intensive_supply_1km_animal_matrix, np.nansum(x.value[x.value > 0])
