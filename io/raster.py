"""
raster.py

栅格数据读写与处理模块
Raster data reading, writing, and processing module
"""
import rasterio as rio
import numpy as np

def read_raster(filepath, is_meta=False):
    """
    读取栅格数据。
    Read raster data from file.
    """
    with rio.open(filepath) as dst:
        arr = dst.read(1)
        arr = np.where(arr == dst.nodata, 0, arr).astype(np.float32)
        if is_meta:
            return arr, dst.profile
        return arr

def write_raster(filepath, data, profile):
    """
    写入栅格数据。
    Write raster data to file.
    """
    with rio.open(filepath, "w", **profile) as dst:
        dst.write(data, 1) 