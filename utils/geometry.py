"""
geometry.py

空间分析与掩膜生成工具模块
Spatial analysis and mask generation utilities module
"""
import numpy as np
import rasterio.features

def polygon_to_mask(polygons, shape, transform=None, dtype=np.uint8):
    """
    将多边形转为掩膜。
    Convert polygons to mask array.
    """
    mask = rasterio.features.rasterize(
        [(geom, 1) for geom in polygons],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=dtype
    )
    return mask 