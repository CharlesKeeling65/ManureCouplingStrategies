"""
logging.py

日志与进度条等通用工具模块
Logging, progress bar, and other general utilities module
"""
import logging

def setup_logging(logfile=None, level=logging.INFO):
    """
    配置日志系统。
    Setup logging system.
    """
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    ) 