# Manure Coupling Strategies 算法库

## 目录结构与功能说明（中英文）

```mermaid
graph TD
  A[ManureCouplingStrategies] --> B(core)
  A --> C(io)
  A --> D(utils)
  B --> B1[transport.py]
  B --> B2[redistribution.py]
  C --> C1[raster.py]
  C --> C2[vector.py]
  D --> D1[geometry.py]
  D --> D2[logging.py]
```

---

## 1. `core/` 核心算法 Core Algorithms

- **transport.py**  
  粪便运输空间分配算法（如线性规划、距离约束等）。  
  Implements manure transport spatial allocation algorithms (e.g., linear programming, distance constraints).
  - `optimal_allocation_linprog`：主分配函数，输入余缺栅格和最大运输距离，输出分配后余缺栅格。
    - Main allocation function, input: supply-demand raster and max transport distance, output: allocated raster.
  - 其他辅助函数如`direct_list`、`move_array`等。
    - Other helpers: `direct_list`, `move_array`, etc.

- **redistribution.py**  
  畜禽再分配优化算法（如按行政区、畜种、线性规划等）。  
  Implements livestock redistribution optimization (by region, species, linear programming, etc.).
  - `update_inPs`：全局/分区再分配主函数。
    - Main function for global/region-wise redistribution.
  - `relocate`：按需求优先分配。
    - Priority-based allocation by demand.

---

## 2. `io/` 数据读写与预处理 Data I/O & Preprocessing

- **raster.py**  
  栅格数据的读取、写入、掩膜等操作。  
  Raster data reading, writing, masking, etc.
- **vector.py**  
  矢量数据（如行政区shp）的读取、处理。  
  Vector data (e.g., administrative boundaries) reading and processing.

---

## 3. `utils/` 工具函数 Utilities

- **geometry.py**  
  空间分析、掩膜生成等。  
  Spatial analysis, mask generation, etc.
- **logging.py**  
  日志、进度条等通用工具。  
  Logging, progress bar, and other general utilities.

---

## 4. 主要函数接口示例 Example Main Function Interfaces

```python
# core/transport.py
def optimal_allocation_linprog(sd_raster, distance, ...):
    """主分配函数 Main allocation function for manure transport."""

# core/redistribution.py
def update_inPs(intensive_supply_1km_animal_matrix, adm_shp, adm_code_list, ...):
    """主函数 Main function for livestock redistribution."""
```

---

## 设计说明 Design Notes

- **高内聚、低耦合 High Cohesion, Low Coupling**：核心算法、数据I/O、工具函数分离，便于维护和扩展。
  - Core algorithms, data I/O, and utilities are separated for maintainability and extensibility.
- **接口清晰 Clear Interfaces**：每个模块只暴露必要的主函数，便于主流程调用。
  - Each module exposes only necessary main functions for easy integration.
- **易于测试 Easy to Test**：可为每个模块单独编写单元测试。
  - Each module can be unit tested independently.

---

如需详细函数参数与用法，请参考各模块内的文档字符串和注释。
For detailed function parameters and usage, please refer to the docstrings and comments in each module.
