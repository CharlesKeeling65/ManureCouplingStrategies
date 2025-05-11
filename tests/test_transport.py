import numpy as np
from core.transport import optimal_allocation_linprog

def test_optimal_allocation_linprog_simple():
    # 构造一个简单的供需栅格
    # 2x2: 左上+10, 右下-10, 其余为0
    sd_raster = np.array([[10, 0], [0, -10]], dtype=np.float32)
    distance = 1
    result = optimal_allocation_linprog(sd_raster, distance)
    # 由于只有一对供需，最优分配后应全为0
    assert np.allclose(result, np.zeros_like(sd_raster), atol=1e-4)

def test_optimal_allocation_linprog_no_supply():
    # 全为负，无法分配
    sd_raster = np.array([[0, 0], [0, -10]], dtype=np.float32)
    distance = 1
    result = optimal_allocation_linprog(sd_raster, distance)
    assert np.allclose(result, sd_raster, atol=1e-4)

def test_optimal_allocation_linprog_no_demand():
    # 全为正，无法分配
    sd_raster = np.array([[10, 0], [0, 0]], dtype=np.float32)
    distance = 1
    result = optimal_allocation_linprog(sd_raster, distance)
    assert np.allclose(result, sd_raster, atol=1e-4) 