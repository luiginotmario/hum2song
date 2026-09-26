"""Logging setup shared by the download, train, and eval scripts."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return the named logger."""
    return logging.getLogger(name)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once for a CLI process."""
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=resolved, format=LOG_FORMAT)
