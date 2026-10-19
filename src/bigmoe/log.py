import logging
import os
import sys

logger = logging.getLogger("bigmoe")


def _setup():
    # leave logging alone if the application configured it
    if logger.handlers or logging.getLogger().handlers:
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("[bigmoe] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.WARNING if os.environ.get("BIGMOE_QUIET") else logging.INFO)
    logger.propagate = False


_setup()


def info(msg, *args):
    logger.info(msg, *args)


def warn(msg, *args):
    logger.warning(msg, *args)
