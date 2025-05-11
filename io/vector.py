"""
vector.py

矢量数据读写与处理模块
Vector data reading, writing, and processing module
"""
import geopandas as gpd

def read_vector(filepath):
    """
    读取矢量数据。
    Read vector data from file.
    """
    return gpd.read_file(filepath)

def write_vector(filepath, data):
    """
    写入矢量数据。
    Write vector data to file.
    """
    data.to_file(filepath) 