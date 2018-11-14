from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from .log_cfg import logger, unconfigure_logging, enable_logging
