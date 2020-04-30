# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-20 18:50:13
# @Last Modified: 2020-04-30 17:14:24
# ------------------------------------------------------------------------------ #
# Callable in your scripts as e.g. `cov.plot.timeseries()`
# Plot functions and helper classes
# Design ideas:
# * Global Parameter Object?
#   - Maybe only for defaults of function parameters but
#   - Have functions be solid stand-alone and only set kwargs from "rc_params"
# * keep model, trace, ax as the three first arguments for every function
# ------------------------------------------------------------------------------ #

import logging
import datetime
import locale
import copy

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------ #
# Plotting functions
# ------------------------------------------------------------------------------ #


def _timeseries(x, y, ax=None, what="data", draw_ci_95=None, draw_ci_75=None, **kwargs):
    """
        low-level function to plot anything that has a date on the x-axis.

        x : array of datetime.datetime
            times for the x axis

        y : array, 1d or 2d
            data to plot. if 2d, we plot the CI as fill_between (if CI enabled in rc
            params)
            if 2d, then first dim is realization and second dim is time matching `x`
            if 1d then first tim is time matching `x`

        ax : mpl axes element, optional
            plot into an existing axes element. default: None

        what : str, optional
            what type of data is provided in x. sets the style used for plotting:
            * `data` for data points
            * `fcast` for model forecast (prediction)
            * `model` for model reproduction of data (past)

        kwargs : dict, optional
            directly passed to plotting mpl.

    """

    # ------------------------------------------------------------------------------ #
    # Default parameter
    # ------------------------------------------------------------------------------ #

    if draw_ci_95 is None:
        draw_ci_95 = rcParams["draw_ci_95"]

    if draw_ci_75 is None:
        draw_ci_75 = rcParams["draw_ci_75"]

    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 3))

    if x.shape[0] != y.shape[-1]:
        log.exception(f"X rows and y rows do not match: {x.shape[0]} vs {y.shape[0]}")
        raise KeyError("Shape mismatch")

    if y.ndim == 2:
        data = np.median(y, axis=0)
    elif y.ndim == 1:
        data = y
    else:
        log.exception(f"y needs to be 1 or 2 dimensional, but has shape {y.shape}")
        raise KeyError("Shape mismatch")

    # ------------------------------------------------------------------------------ #
    # kwargs
    # ------------------------------------------------------------------------------ #

    if what is "data":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_data"])
        if "marker" not in kwargs:
            # this needs fixing.
            kwargs = dict(kwargs, marker="d")
    elif what is "fcast":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs:
            kwargs = dict(kwargs, ls="--")
    elif what is "model":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs:
            kwargs = dict(kwargs, ls="-")

    # ------------------------------------------------------------------------------ #
    # plot
    # ------------------------------------------------------------------------------ #
    ax.plot(x,
        data,
        label="Data",
        **kwargs)

    if "linewidth" in kwargs:
        del kwargs["linewidth"]
    if "marker" in kwargs:
        del kwargs["marker"]
    kwargs["lw"] = 0

    # set alpha to less than 1
    if draw_ci_95 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=2.5, axis=0),
            np.percentile(y, q=97.5, axis=0),
            alpha=0.1,
            **kwargs,
        )

    if draw_ci_75 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=12.5, axis=0),
            np.percentile(y, q=87.5, axis=0),
            alpha=0.1,
            **kwargs,
        )

    return ax


def _get_array_from_trace_via_date(model, trace_var, dates):
    #This could be wrong!
    indices = (dates - model.data_begin).days
    # i would really like to always have the 0 index present, even when no bundeslaender
    print(f"data with ndim {trace_var.ndim}")
    if trace_var.ndim == 2:
        return trace_var[:, :, indices]
    else:
        return trace_var[:, indices]


# ------------------------------------------------------------------------------ #
# Parameters, we have to do this first so we can have default arguments
# ------------------------------------------------------------------------------ #


def get_rcparams_default():
    """
        Get a Param (dict) of the default parameters.
        Here we set our default values. Assigned once to module variable
        `rcParamsDefault` on load.
    """
    par = Param(
        locale="en_US",
        date_format="%b %-d",
        date_show_minor_ticks=True,
        rasterization_zorder=-1,
        draw_ci_95=True,
        draw_ci_75=False,
        color_model="tab:green",
        color_data="tab:blue",
        color_annot="#646464",
    )

    return par


def set_rcparams(par):
    """
        Set the rcparameters used for plotting. provided instance of `Param` has to have the following keys (attributes).

        Attributes
        ----------
        locale : str
            region settings, passed to `setlocale()`. Default: "en_US"

        date_format : str
            Format the date on the x axis of time-like data (see https://strftime.org/)
            example April 1 2020:
            "%m/%d" 04/01, "%-d. %B" 1. April
            Default "%b %-d", becomes April 1

        date_show_minor_ticks : bool
            whether to show the minor ticks (for every day). Default: True

        rasterization_zorder : int or None
            Rasterizes plotted content below this value, set to None to keep everything
            a vector, Default: -1

        draw_ci_95 : bool
            For timeseries plots, indicate 95% Confidence interval via fill between.
            Default: True

        draw_ci_75 : bool,
            For timeseries plots, indicate 75% Confidence interval via fill between.
            Default: False

        color_model : str,
            Base color used for model plots, mpl compatible color code "C0", "#303030"
            Default : "tab:green"

       color_patalot : str,
            Base color used for data
            Default : "tab:blue"

        color_annot : str,
            Color to use for annotations
            Default : "#646464"

        Example
        ------
        ```
        pars = cov.plot.get_rcparams_default()
        pars["locale"]="de_DE"
        cov.plot.set_rcparams(pars)
        ```
    """
    for key in get_rcparams_default().keys():
        assert key in par.keys(), "Provide all keys that are in .get_rcparams_default()"

    global rcParams
    rcParams = copy.deepcopy(par)


class Param(dict):
    """
        Paramters Base Class (a tweaked dict)

        We inherit from dict and also provide keys as attributes, mapped to `.get()` of
        dict. This avoids the KeyError: if getting parameters via `.the_parname`, we
        return None when the param does not exist.

        Avoid using keys that have the same name as class functions etc.

        Example
        -------
        ```
        foo = Param(lorem="ipsum")
        print(foo.lorem)
        >>> 'ipsum'
        print(foo.does_not_exist is None)
        >>> True
        ```
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return Param(copy.deepcopy(dict(self), memo=memo))

    @property
    def varnames(self):
        return [*self]


# ------------------------------------------------------------------------------ #
# Formatting helpers
# ------------------------------------------------------------------------------ #

# format yaxis 10_000 as 10 k
format_k = lambda num, _: "${:.0f}\,$k".format(num / 1_000)


def _format_date_xticks(ax, minor=None):
    locale.setlocale(locale.LC_ALL, rcParams.locale+".UTF-8")#We have to utf-8 here atleast on my pc. I dont know if i
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(interval=1, byweekday=matplotlib.dates.SU)
    )
    if rcParams["date_show_minor_ticks"]:
        ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(rcParams["date_format"]))


def truncate_number(number, precision):
    return "{{:.{}f}}".format(precision).format(number)


def print_median_CI(arr, prec=2):
    f_trunc = lambda n: truncate_number(n, prec)
    med = f_trunc(np.median(arr))
    perc1, perc2 = (
        f_trunc(np.percentile(arr, q=2.5)),
        f_trunc(np.percentile(arr, q=97.5)),
    )
    return "Median: {}\nCI: [{}, {}]".format(med, perc1, perc2)


def conv_time_to_mpl_dates(arr):
    try:
        return matplotlib.dates.date2num(
            [datetime.timedelta(days=float(date)) + date_begin_sim for date in arr]
        )
    except:
        return matplotlib.dates.date2num(
            datetime.timedelta(days=float(arr)) + date_begin_sim
        )


# ------------------------------------------------------------------------------ #
# init
# ------------------------------------------------------------------------------ #
# set global parameter variables
rcParams = get_rcparams_default()
