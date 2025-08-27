import logging
import os
import sys

try:
    import colorlog
except ImportError:
    colorlog = None

def get_logger(name=__name__, log_level=logging.INFO, log_file=None, use_color=True):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        if use_color and colorlog:
            formatter = colorlog.ColoredFormatter(
                "%(log_color)s[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
                datefmt="%H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                }
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
        else:
            formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)

    return logger
