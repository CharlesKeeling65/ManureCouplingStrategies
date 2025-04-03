"""
ManureTransport 库基本使用示例
"""

import numpy as np
from ManureTransport.core.transport import SpeciesAverageTransportModel
from ManureTransport.core.spatial import SimpleDistanceStrategy
from ManureTransport.utils.logging_utils import setup_logging
from ManureTransport.utils.config import get_default_species_config

# 设置日志
setup_logging()

# 创建模拟数据
supply = np.array([[100, 150, 0], [200, 0, 50], [0, 75, 125]])

demand = np.array([[50, 75, 100], [60, 80, 70], [90, 40, 30]])

# 使用默认物种配置
species_config = get_default_species_config()

# 创建距离策略
distance_strategy = SimpleDistanceStrategy(max_distance=10)  # 假设单位为公里

# 创建运输模型
model = SpeciesAverageTransportModel(
    supply_data=supply,
    demand_data=demand,
    distance_strategy=distance_strategy,
    species_config=species_config,
)

# 运行模型
result = model.solve()

# 输出结果
print("分配结果矩阵:")
print(result.allocation)
print("\n供给满足率:")
print(result.supply_satisfaction)
print("\n需求满足率:")
print(result.demand_satisfaction)

# 可视化结果
# 注意：这里只是示例代码，实际的可视化函数需要在库中实现
# result.visualize_flow()
