"""
ManureTransport - 粪便养分运输优化和分配库

这个Python库提供了用于粪便养分供需运输优化的工具，包括：
- 基于距离的运输优化
- 多种距离策略的实现
- 养分供需数据处理工具
"""

__version__ = "0.1.0"
__author__ = "wangyb"

# 导入核心功能
from ManureTransport.core.spatial import (
    DistanceStrategy,
    UnlimitedDistanceStrategy,
    SimpleDistanceStrategy,
)

from ManureTransport.core.transport import (
    TransportModel,
    UnlimitedTransportModel,
    SpeciesAverageTransportModel,
)

# 导入IO功能
from ManureTransport.io.data_io import read_raster, write_raster, read_species_data

# 导入工具功能
from ManureTransport.utils.logging_utils import setup_logging
from ManureTransport.utils.config import load_config, get_default_species_config
