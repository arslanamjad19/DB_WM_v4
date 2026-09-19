"""
Logging utilities for the DB-WM v2 codebase.

Mirrors the lightweight, dependency-free logging style of the reference
``funcobspy`` package (see ``funcobspy/functionobservers/log_utils.py``) so the
two codebases feel consistent, but drops the UNIX ``SysLogHandler`` requirement
that makes the original brittle on non-UNIX / containerised research machines.
"""
import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logger(level: str = "INFO", name: str | None = None) -> logging.Logger:
    """
    Configure and return a module-level logger.

    :param level: logging level string, one of
                  ``['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']``.
    :param name: name passed to :func:`logging.getLogger`. If ``None`` the
                 calling module's ``__name__`` is used.
    :return: a configured :class:`logging.Logger` instance.
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = level.upper()
    if level not in level_map:
        print("ERROR: Invalid value {} for the logging level.".format(level))
        level = "INFO"

    logging.basicConfig(level=level_map[level], format=LOG_FORMAT)
    logger_out = logging.getLogger(name if isinstance(name, str) else __name__)
    logger_out.setLevel(level_map[level])
    return logger_out


def check_pos_int(v) -> bool:
    """
    Return ``True`` iff ``v`` can be cast to a strictly positive integer.

    :param v: candidate value.
    :return: boolean status.
    """
    try:
        return int(v) > 0
    except (ValueError, TypeError):
        return False


def check_pos_float(v) -> bool:
    """
    Return ``True`` iff ``v`` can be cast to a strictly positive float.

    :param v: candidate value.
    :return: boolean status.
    """
    try:
        return float(v) > 0.0
    except (ValueError, TypeError):
        return False
