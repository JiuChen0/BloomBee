import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        name: str = "BloomBee",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    ):
        if self._initialized:
            return

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(fmt)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self._initialized = True

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def debug(self, message):
        self.logger.debug(message)

    @classmethod
    def get_logger(cls):
        """Get the logger instance"""
        if not cls._instance:
            cls()
        return cls._instance.logger


GLOBAL_LOGGER = None


def get_logger():
    global GLOBAL_LOGGER
    if GLOBAL_LOGGER is None:
        GLOBAL_LOGGER = Logger()
    return GLOBAL_LOGGER
