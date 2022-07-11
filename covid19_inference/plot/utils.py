import locale
import matplotlib as mpl
import numpy as np
import pandas as pd
from .rcParams import *


def add_watermark(ax, mark="Dehning et al. 10.1126/science.abb9789"):
    """
    Add our arxive url to an axes as (upper right) title
    """

    # fig.text(
    #     pos[0],
    #     pos[1],
    #     "Dehning et al.",
    #     fontsize="medium",
    #     transform=  fig.transFigure,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    #     color="#646464"
    #     # bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    # )

    ax.set_title(mark, fontsize="small", loc="right", color="#646464")


def format_k(prec):
    """
    format yaxis 10_000 as 10 k.
    _format_k(0)(1200, 1000.0) gives "1 k"
    _format_k(1)(1200, 1000.0) gives "1.2 k"
    """

    def inner(xval, tickpos):
        if xval == 0:
            return "0"
        else:
            return f"${xval/1_000:.{prec}f}\,$k"

    return inner


def format_date_xticks(ax, minor=None):
    # ensuring utf-8 helps on some setups
    locale.setlocale(locale.LC_ALL, rcParams.locale + ".UTF-8")
    ax.xaxis.set_major_locator(
        mpl.dates.WeekdayLocator(interval=1, byweekday=mpl.dates.SU)
    )
    if minor is None:
        # overwrite local argument with rc params only if default.
        minor = rcParams["date_show_minor_ticks"]
    if minor is True:
        ax.xaxis.set_minor_locator(mpl.dates.DayLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(rcParams["date_format"]))


def get_array_from_idata(idata, var, from_type="posterior"):
    """Reshapes and returns an numpy array from an arviz idata

    Parameters
    ----------
    idata : :class:`arviz.InferenceData`
        InferenceData object
    var : str
        Variable name
    from_type : str, optional
        Type of data to return. Options are:
        * `posterior` : posterior samples
        * `prior` : prior samples
        * ... check idata attributes for options

    Returns
    -------
    array : numpy.ndarray with chain and smaples flattened
    """

    var = np.array(getattr(idata, from_type)[var])
    if from_type == "predictions":
        var = var.reshape((var.shape[0] * var.shape[1],) + var.shape[2:])
    var = var.reshape((var.shape[0] * var.shape[1],) + var.shape[2:])

    # Remove nans (normally there are 0 nans but can happen if you use where operations)
    var = var[~np.isnan(var).any(tuple(range(1, var.ndim))), ...]
    return var


def get_array_from_idata_via_date(model, idata, var, start=None, end=None, dates=None):
    """
    Parameters
    ----------
    model : :class:`Cov19Model`

    idata : :class:`arviz.InferenceData`

    var : str
        the variable name in the trace

    start : datetime.datetime
        get all data for a range from `start` to `end`. (both boundary
        dates included)

    end : datetime.datetime

    dates : list of datetime.datetime objects, optional
        the dates for which to get the data. Default: None, will return
        all available data.

    Returns
    -------
    data : nd array, 3 dim
        the elements from the trace matching the dates.
        dimensions are as follows
        0 samples, if no samples only one entry
        1 data with time matching the returned `dates` (if compatible variable)
        2 region, if no regions only one entry

    dates : pandas DatetimeIndex
        the matching dates. this is essnetially an array of dates than can be passed
        to matplotlib

    Example
    -------
    .. code-block::

        import covid19_inference as cov
        model, trace = cov.create_example_instance()
        y, x = cov.plot._get_array_from_trace_via_date(
            model, trace, "lambda_t", model.data_begin, model.data_end
        )
        ax = cov.plot._timeseries(x, y[:,:,0], what="model")
    """

    ref = model.sim_begin
    # the variable `new_cases` and some others (?) used to have different bounds
    # 20-05-27: not anymore, we made things consistent. let's remove this at some point
    # if "new_cases" in var:
    # ref = model.data_begin

    if dates is None:
        if start is None:
            start = ref
        if end is None:
            end = model.sim_end
        dates = pd.date_range(start=start, end=end)
    else:
        assert start is None and end is None, "do not pass start/end with dates"
        # make sure its the right format
        dates = pd.DatetimeIndex(dates)

    indices = (dates - ref).days

    assert var in idata.posterior, "var should be in trace.varnames"
    assert np.all(indices >= 0), (
        "all dates should be after the model.sim_begin "
        + "(note that `new_cases` start at model.data_begin)"
    )
    assert np.all(indices < model.sim_len), "all dates should be before model.sim_end"

    trace = get_array_from_idata(idata, var, "posterior")

    # here we make sure that the returned array always has the same dimensions:
    if trace.ndim == 3:
        ret = trace[:, indices, :]
    elif trace.ndim == 2:
        ret = trace[:, indices]
        # ret = trace[:, indices, None]
        # 2020-05-06: jd and ps decided not to pad dimensions, not sure if it is more
        # confusing to have changing dimensions or dimensions that are not needed
        # in case of the non-hierarchical model
        # to access the region if you are not sure if it exists use an ellipsis:
        # region = ...
        # trace[var][:, indices, region]
        # will work fine if trace[var] is only 2-dimensional

    return ret, dates
