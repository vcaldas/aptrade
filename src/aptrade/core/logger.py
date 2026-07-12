"""
Logging module for APTRADE FastAPI application.

This module provides a centralized logging solution for the FastAPI application.
It configures appropriate loggers for the FastAPI server environment.
"""

import datetime as dt
import logging
import os
from zoneinfo import ZoneInfo

from .config import settings

VALID_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Global logger instance
_logger: logging.Logger | None = None


class TimezoneFormatter(logging.Formatter):
    """Formatter that renders timestamps in an explicit timezone."""

    def __init__(self, fmt: str, timezone_name: str) -> None:
        super().__init__(fmt)
        self._tz = ZoneInfo(timezone_name)

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt_value = dt.datetime.fromtimestamp(record.created, tz=self._tz)
        if datefmt:
            return dt_value.strftime(datefmt)
        return dt_value.isoformat(timespec="seconds")


def get_logger() -> logging.Logger:
    """
    Get the configured logger instance.

    This function ensures the logger is configured only once and reused across
    all modules. It uses the uvicorn logger by default for FastAPI compatibility.

    Returns:
        logging.Logger: The configured logger instance
    """
    global _logger

    if _logger is None:
        # Use an application logger by default to avoid uvicorn-specific handler behavior.
        logger_name = os.environ.get("LOGGING_LOGGER_NAME", "aptrade")
        _logger = logging.getLogger(logger_name)

        # Set level from environment
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        log_level = VALID_LEVELS.get(log_level_str, logging.INFO)

        timezone_name = os.environ.get("LOG_TIMEZONE", settings.APP_TIMEZONE)

        has_stream_handler = any(
            isinstance(handler, logging.StreamHandler) for handler in _logger.handlers
        )
        if not has_stream_handler:  # type: ignore[attr-defined]
            handler = logging.StreamHandler()
            # use a minimal default formatter similar to uvicorn's default
            handler.setFormatter(
                TimezoneFormatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    timezone_name=timezone_name,
                )
            )
            _logger.addHandler(handler)

        # Keep levels in sync so INFO/DEBUG logs are visible in terminal as expected.
        for handler in _logger.handlers:
            handler.setLevel(log_level)

        _logger.setLevel(log_level)
        _logger.disabled = False

        # Optionally force propagation (useful in some testing setups)
        force_propagate = os.environ.get("LOGGING_FORCE_PROPAGATE", "").lower()
        if force_propagate in ("1", "true", "yes"):
            _logger.propagate = True

    return _logger
