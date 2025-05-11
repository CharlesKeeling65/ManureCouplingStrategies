import numpy as np
from shapely.geometry import box
from utils.geometry import polygon_to_mask
from utils.logging import setup_logging
import logging

def test_polygon_to_mask():
    polygons = [box(0, 0, 1, 1)]
    shape = (2, 2)
    mask = polygon_to_mask(polygons, shape)
    assert mask.shape == shape
    assert mask.max() == 1

def test_setup_logging(capsys):
    setup_logging()
    logging.info("test message")
    captured = capsys.readouterr()
    assert "test message" in captured.out or "test message" in captured.err 