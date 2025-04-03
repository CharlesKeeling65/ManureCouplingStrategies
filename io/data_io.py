"""
数据IO模块，提供数据读写功能
"""

import numpy as np
import rasterio as rio
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union


def read_raster(
    file_path: Union[str, Path], no_data_value: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    读取栅格数据文件

    Args:
        file_path: 栅格文件路径
        no_data_value: 如果提供，将使用此值替换NoData值，否则使用文件中的NoData值

    Returns:
        包含栅格数据的NumPy数组和元数据字典的元组
    """
    with rio.open(file_path) as src:
        data = src.read(1)
        meta = src.meta.copy()

        # 处理NoData值
        if no_data_value is not None:
            data = np.where(data == src.nodata, no_data_value, data).astype(np.float32)
        else:
            data = np.where(data == src.nodata, 0, data).astype(np.float32)

        return data, meta


def write_raster(
    data: np.ndarray,
    meta: Dict[str, Any],
    file_path: Union[str, Path],
    no_data_mask: Optional[np.ndarray] = None,
) -> None:
    """
    将NumPy数组写入栅格文件

    Args:
        data: 要写入的NumPy数组数据
        meta: 栅格元数据
        file_path: 输出文件路径
        no_data_mask: 如果提供，此掩码中为True的位置将被设置为NoData值
    """
    # 确保输出目录存在
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # 如果提供了掩码，将掩码区域设置为NoData
    if no_data_mask is not None:
        data = np.where(no_data_mask, meta["nodata"], data)

    # 写入数据
    with rio.open(file_path, "w", **meta) as dst:
        dst.write(data, 1)


def read_species_data(
    species_config: Dict[str, Dict[str, Any]],
    base_path: Union[str, Path],
    year: str = "2021",
    resolution: str = "1km",
) -> Dict[str, np.ndarray]:
    """
    读取所有物种的供应数据

    Args:
        species_config: 物种配置字典
        base_path: 基础路径
        year: 年份
        resolution: 分辨率

    Returns:
        包含每个物种供应数据的字典
    """
    base_path = Path(base_path)
    species_data = {}

    for species in species_config:
        file_path = (
            base_path
            / f"output_supply_demand/supply/{species}_fer_{year}_{resolution}.tif"
        )
        data, _ = read_raster(file_path)
        species_data[species] = data

    return species_data
