import sys

import logging

logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))

_is_interactive = bool(getattr(sys, 'ps1', sys.flags.interactive))
_hander = None


def unconfigure_logging():
    if _hander is not None:
        logger.removeHandler(_hander)

    logger.setLevel(logging.NOTSET)


def enable_logging(verbosity=logging.ERROR, stream=sys.stderr, format=logging.BASIC_FORMAT):
    unconfigure_logging()

    logger.setLevel(verbosity)
    _handler = logging.StreamHandler(stream)
    _handler.setFormatter(logging.Formatter(format, None))
    logger.addHandler(_handler)


# If we are in an interactive environment (like Jupyter), set loglevel to INFO and pipe the output to stdout.
if _is_interactive:
    enable_logging(logging.INFO, sys.stdout)
else:
    enable_logging(logging.WARNING, sys.stderr)
