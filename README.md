# ManureTransport - 粪便养分运输优化和分配库

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.6%2B-blue.svg)

## 项目简介

ManureTransport 是一个专为优化粪便养分运输和分配而设计的 Python 库。该库提供了一套完整的工具，用于解决农业领域中养殖场粪便资源向种植区的合理输送和分配问题，帮助农业生产实现资源的高效循环利用。

## 主要功能

- **空间分析**：提供多种距离策略和空间匹配算法
- **运输优化**：基于不同约束条件的运输模型实现
- **数据处理**：栅格数据读写、物种配置数据处理等功能
- **可视化**：运输流向和养分分配结果的可视化支持

## 安装方法

```bash
pip install ManureTransport
```

## 快速入门

### 基本使用

```python
from ManureTransport.core.transport import SpeciesAverageTransportModel
from ManureTransport.core.spatial import SimpleDistanceStrategy
from ManureTransport.io.data_io import read_raster, write_raster
from ManureTransport.utils.config import load_config

# 加载配置
config = load_config('config.yaml')

# 读取供需数据
supply_data = read_raster('supply.tif')
demand_data = read_raster('demand.tif')

# 创建距离策略
distance_strategy = SimpleDistanceStrategy(max_distance=50)  # 单位：公里

# 创建运输模型
transport_model = SpeciesAverageTransportModel(
    supply_data=supply_data,
    demand_data=demand_data,
    distance_strategy=distance_strategy,
    config=config
)

# 运行模型
result = transport_model.solve()

# 保存结果
write_raster('allocation_result.tif', result.allocation)
```

### 不同距离策略

```python
# 不限制距离的策略
from ManureTransport.core.spatial import UnlimitedDistanceStrategy
unlimited_strategy = UnlimitedDistanceStrategy()

# 自定义距离策略
from ManureTransport.core.spatial import DistanceStrategy

class CustomDistanceStrategy(DistanceStrategy):
    def calculate_distance(self, source, target):
        # 自定义距离计算逻辑
        pass
        
    def is_within_limit(self, distance):
        # 自定义距离限制判断逻辑
        pass
```

## API文档

### 核心模块

#### 空间分析模块 (core.spatial)

- `DistanceStrategy`: 距离策略基类，提供距离计算接口
- `UnlimitedDistanceStrategy`: 无限距离策略，不设置最大运输距离限制
- `SimpleDistanceStrategy`: 简单距离策略，基于欧氏距离设定最大运输距离

#### 运输模块 (core.transport)

- `TransportModel`: 运输模型基类
- `UnlimitedTransportModel`: 无限制运输模型
- `SpeciesAverageTransportModel`: 基于物种平均值的运输模型

### IO 模块

- `read_raster`: 读取栅格数据
- `write_raster`: 写入栅格数据
- `read_species_data`: 读取物种参数配置数据

### 工具模块

- `setup_logging`: 配置日志系统
- `load_config`: 加载配置文件
- `get_default_species_config`: 获取默认物种配置

## 贡献指南

欢迎对本项目进行贡献！请遵循以下步骤:

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 将您的更改推送到分支 (`git push origin feature/amazing-feature`)
5. 创建一个 Pull Request

## 开源协议

本项目基于 MIT 协议开源 - 详见 [LICENSE](LICENSE) 文件
