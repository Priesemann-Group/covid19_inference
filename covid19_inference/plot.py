# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-20 18:50:13
# @Last Modified: 2020-04-21 09:56:52
# ------------------------------------------------------------------------------ #
# Callable in your scripts as e.g. `cov.plot.timeseries()`
# Plot functions and helper classes
# Design ideas:
# * Global Parameter Object?
#   - Maybe only for defaults of function parameters but
#   - Have functions be solid stand-alone and only set kwargs from "rc_params"
#
# ------------------------------------------------------------------------------ #

import logging
import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


def timeseries():
    pass
    log.info("test")
