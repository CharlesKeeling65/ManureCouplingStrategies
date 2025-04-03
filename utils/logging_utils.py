"""
日志工具模块
"""

import logging
import datetime
from pathlib import Path


def setup_logging(
    log_dir: Path = Path("../log"),
    prefix: str = "manure_transport",
    resolution: str = "1km",
    level: int = logging.INFO,
) -> Path:
    """
    设置日志记录

    Args:
        log_dir: 日志目录
        prefix: 日志文件前缀
        resolution: 分辨率字符串
        level: 日志级别

    Returns:
        日志文件路径
    """
    # 确保日志目录存在
    log_dir.mkdir(exist_ok=True, parents=True)

    # 创建日志文件路径
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_file = log_dir / f"{prefix}_{resolution}_{timestamp}.log"

    # 配置日志
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return log_file
