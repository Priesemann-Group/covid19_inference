__version__ = "unknown"
from ._version import __version__

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

import filelock

logging.getLogger("filelock").setLevel(logging.ERROR)

# from .data_retrieval import GOOGLE
from . import data_retrieval
from .plot import *
from . import model
from .model import Cov19Model
from .sampling import robust_sample
from . import sampling

from ._dev_helper import create_example_instance
