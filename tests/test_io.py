import numpy as np
import geopandas as gpd
import tempfile
import os
from io.raster import read_raster, write_raster
from io.vector import read_vector, write_vector

def test_raster_io(tmp_path):
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'width': 2,
        'height': 2,
        'crs': None,
        'transform': (1, 0, 0, 0, -1, 0),
        'nodata': 0
    }
    f = tmp_path / 'test.tif'
    write_raster(f, arr, profile)
    arr2 = read_raster(f)
    assert np.allclose(arr, arr2)

def test_vector_io(tmp_path):
    gdf = gpd.GeoDataFrame({'a': [1]}, geometry=gpd.points_from_xy([0], [0]))
    f = tmp_path / 'test.shp'
    write_vector(f, gdf)
    gdf2 = read_vector(f)
    assert gdf2.shape[0] == 1 