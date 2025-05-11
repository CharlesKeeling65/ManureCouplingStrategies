import numpy as np
from core.redistribution import relocate, update_inPs

def test_relocate_simple():
    S = np.array([[10, 0], [0, 0]], dtype=np.float32)
    D = np.array([[0, 0], [0, 10]], dtype=np.float32)
    S_new = relocate(S, D)
    # 供给应被分配到需求最大的位置
    assert S_new[1, 1] == 10
    assert S_new[0, 0] == 0

def test_update_inPs_shape():
    # 只测试形状和类型
    mat = np.ones((2, 2, 2), dtype=np.float32)
    class DummyShp:
        geometry = [None]
    out, delta = update_inPs(mat, DummyShp(), [1])
    assert out.shape == mat.shape
    assert isinstance(delta, (int, float)) 