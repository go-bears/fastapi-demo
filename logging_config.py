"""
This module is responsible for setting up the logging for the entire application.
It uses the loguru library to handle logging, and it's configured to log to a file, stdout, and stderr.

log files are stored in the logs/ directory.
"""

from loguru import logger
import sys


def setup_logging():
    logger.add(
        "logs/fimio-logs-{time}.log", rotation="100 MB", compression="zip", enqueue=True
    )
    logger.add(
        sys.stdout,
        format="{time} {level} {message}",
        filter="info",
        level="INFO",
        colorize=True,
    )
    logger.add(
        sys.stdout,
        format="{time} {level} {message}",
        filter="error",
        level="ERROR",
        colorize=True,
    )
    logger.add(
        sys.stdout,
        format="{time} {level} {message}",
        filter="debug",
        level="DEBUG",
        colorize=True,
    )


if __name__ == "__main__":
    """
    This is a test of the logging system.
    $ python logging_config.py
    """
    setup_logging()
    logger.info("Hello, world! Success")
    logger.error("This is an error message")
    logger.debug("This is a debug message")
