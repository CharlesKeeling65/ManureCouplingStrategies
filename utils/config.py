"""
配置工具模块
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    从YAML文件加载配置

    Args:
        config_file: 配置文件路径

    Returns:
        配置字典
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_default_species_config() -> Dict[str, Dict[str, int]]:
    """
    获取默认的物种配置

    Returns:
        默认物种配置字典
    """
    return {
        "pig": {"distance": 10},
        "beef": {"distance": 10},
        "dairy": {"distance": 10},
        "SG": {"distance": 20},
        "poultry": {"distance": 40},
    }
