import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plt

from .rcParams import *
from .utils import *

log = logging.getLogger(__name__)



# ------------------------------------------------------------------------------ #
# Time series plotting functions
# ------------------------------------------------------------------------------ #


def timeseries_overview(
    model,
    idata,
    start=None,
    end=None,
    region=None,
    color=None,
    save_to=None,
    offset=0,
    annotate_constrained=True,
    annotate_watermark=True,
    axes=None,
    forecast_label="Forecast",
    forecast_heading=r"$\bf Forecasts\!:$",
    add_more_later=False,
):
    r"""
    Create the time series overview similar to our paper.
    Dehning et al. arXiv:2004.01105
    Contains :math:`\lambda`, new cases, and cumulative cases.

    Parameters
    ----------
    model : :class:`Cov19Model`

    trace : :class:`arviz.InferenceData`
        needed for the data

    offset : int
        offset that needs to be added to the (cumulative sum of) new cases at time
        model.data_begin to arrive at cumulative cases

    start : datetime.datetime
        only used to set xrange in the end
    end : datetime.datetime
        only used to set xrange in the end
    color : str
        main color to use, default from rcParam
    save_to : str or None
        path where to save the figures. default: None, not saving figures
    annotate_constrained : bool
        show the unconstrained constrained annotation in lambda panel
    annotate_watermark : bool
        show our watermark
    axes : np.array of mpl axes
        provide an array of existing axes (from previously calling this function)
        to add more traces. Data will not be added again. Ideally call this first
        with `add_more_later=True`
    forecast_label : str
        legend label for the forecast, default: "Forecast"
    forecast_heading : str
        if `add_more_later`, how to label the forecast section.
        default: "$\bf Forecasts\!:$",
    add_more_later : bool
        set this to true if you plan to add multiple models to the plot. changes the layout (and the color of the fit to past data)

    Returns
    -------
        fig : mpl figure
        axes : np array of mpl axeses (insets not included)

    TODO
    ----
    * Replace `offset` with an instance of data class that should yield the
      cumulative cases. we should not to calculations here.
    """

    figsize = (6, 6)
    # ylim_new = [0, 2_000]
    # ylim_new_inset = [50, 17_000]
    # ylim_cum = [0, 20_000]
    # ylim_cum_inset = [50, 300_000]
    ylim_lam = [-0.15, 0.45]

    label_y_new = f"Daily new\nreported cases"
    label_y_cum = f"Total\nreported cases"
    label_y_lam = f"Effective\ngrowth rate $\lambda^\\ast (t)$"
    label_leg_data = "Data"
    label_leg_dlim = f"Data until\n{model.data_end.strftime('%Y/%m/%d')}"

    if rcParams["locale"].lower() == "de_de":
        label_y_new = f"Täglich neu\ngemeldete Fälle"
        label_y_cum = f"Gesamtzahl\ngemeldeter Fälle"
        label_y_lam = f"Effektive\nWachstumsrate"
        label_leg_data = "Daten"
        label_leg_dlim = f"Daten bis\n{model.data_end.strftime('%-d. %B %Y')}"

    letter_kwargs = dict(x=-0.25, y=1, size="x-large")

    # per default we assume no hierarchical
    if region is None:
        region = ...

    axes_provided = False
    if axes is not None:
        log.debug("Provided axes, adding new content")
        axes_provided = True

    color_data = rcParams.color_data
    color_past = rcParams.color_model
    color_fcast = rcParams.color_model
    color_annot = rcParams.color_annot
    if color is not None:
        color_past = color
        color_fcast = color

    if axes_provided:
        fig = axes[0].get_figure()
    else:
        fig, axes = plt.subplots(
            3,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [2, 3, 3]},
            constrained_layout=True,
        )
        if add_more_later:
            color_past = "#646464"

    if start is None:
        start = model.data_begin
    if end is None:
        end = model.sim_end

    # insets are not reimplemented yet
    insets = []
    insets_only_two_ticks = True
    draw_insets = False

    # ------------------------------------------------------------------------------ #
    # lambda*, effective growth rate
    # ------------------------------------------------------------------------------ #
    ax = axes[0]
    mu = get_array_from_idata(idata,"mu")

    lambda_t, x = get_array_from_idata_via_date(model, idata, "lambda_t")
    y = lambda_t[:, :, region] - mu[...,None]
    _timeseries(x=x, y=y, ax=ax, what="model", color=color_fcast)
    ax.set_ylabel(label_y_lam)
    ax.set_ylim(ylim_lam)

    if not axes_provided:
        ax.text(s="A", transform=ax.transAxes, **letter_kwargs)
        ax.hlines(0, x[0], x[-1], linestyles=":")
        if annotate_constrained:
            try:
                # depending on hierchy delay has differnt variable names.
                # get the shortest one. todo: needs to be change depending on region.
                delay_vars = [var for var in trace.varnames if "delay" in var]
                delay_var = delay_vars.sort(key=len)[0]
                delay = mpl.dates.date2num(model.data_end) - np.percentile(
                    get_array_from_idata(idata,delay_var), q=75
                )
                ax.vlines(delay, -10, 10, linestyles="-", colors=color_annot)
                ax.text(
                    delay + 1.5,
                    0.4,
                    "unconstrained due\nto reporting delay",
                    color=color_annot,
                    horizontalalignment="left",
                    verticalalignment="top",
                )
                ax.text(
                    delay - 1.5,
                    0.4,
                    "constrained\nby data",
                    color=color_annot,
                    horizontalalignment="right",
                    verticalalignment="top",
                )
            except Exception as e:
                log.debug(f"{e}")

    # --------------------------------------------------------------------------- #
    # New cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes[1]

    y_past, x_past = get_array_from_idata_via_date(
        model, idata, "new_cases", model.data_begin, model.data_end
    )
    y_past = y_past[:, :, region]

    y_data = model.new_cases_obs[:, region]

    x_data = pd.date_range(start=model.data_begin, end=model.data_end)

    # data points and annotations, draw only once
    if not axes_provided:
        ax.text(s="B", transform=ax.transAxes, **letter_kwargs)
        _timeseries(
            x=x_data,
            y=y_data,
            ax=ax,
            what="data",
            color=color_data,
            zorder=5,
            label=label_leg_data,
        )
        # model fit
        _timeseries(
            x=x_past,
            y=y_past,
            ax=ax,
            what="model",
            color=color_past,
            label="Fit",
        )
        if add_more_later:
            # dummy element to separate forecasts
            ax.plot(
                [],
                [],
                "-",
                linewidth=0,
                label=forecast_heading,
            )

    # model fcast
    y_fcast, x_fcast = get_array_from_idata_via_date(
        model, idata, "new_cases", model.fcast_begin, model.fcast_end
    )
    y_fcast = y_fcast[:, :, region]
    _timeseries(
        x=x_fcast,
        y=y_fcast,
        ax=ax,
        what="fcast",
        color=color_fcast,
        label=f"{forecast_label}",
    )
    ax.set_ylabel(label_y_new)
    # ax.set_ylim(ylim_new)
    prec = 1.0 / (np.log10(ax.get_ylim()[1]) - 2.5)
    if prec < 2.0 and prec >= 0:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(format_k(int(prec)))
        )

    # ------------------------------------------------------------------------------ #
    # total cases, still needs work because its not in the trace, we cant plot it
    # due to the lacking offset from new to cumulative cases, we cannot calculate
    # either.
    # ------------------------------------------------------------------------------ #

    ax = axes[2]

    y_past, x_past = get_array_from_idata_via_date(
        model, idata, "new_cases", model.data_begin, model.data_end
    )
    y_past = y_past[:, :, region]

    y_data = model.new_cases_obs[:, region]
    x_data = pd.date_range(start=model.data_begin, end=model.data_end)

    x_data, y_data = _new_cases_to_cum_cases(x_data, y_data, "data", offset)
    x_past, y_past = _new_cases_to_cum_cases(x_past, y_past, "trace", offset)

    # data points and annotations, draw only once
    if not axes_provided:
        ax.text(s="C", transform=ax.transAxes, **letter_kwargs)
        _timeseries(
            x=x_data,
            y=y_data,
            ax=ax,
            what="data",
            color=color_data,
            zorder=5,
            label=label_leg_data,
        )
        # model fit
        _timeseries(
            x=x_past,
            y=y_past,
            ax=ax,
            what="model",
            color=color_past,
            label="Fit",
        )
        if add_more_later:
            # dummy element to separate forecasts
            ax.plot(
                [],
                [],
                "-",
                linewidth=0,
                label=forecast_heading,
            )

    # model fcast, needs to start one day later, too. use the end date we got before
    y_fcast, x_fcast = get_array_from_idata_via_date(
        model, idata, "new_cases", model.fcast_begin, model.fcast_end
    )
    y_fcast = y_fcast[:, :, region]

    # offset according to last cumulative model point
    x_fcast, y_fcast = _new_cases_to_cum_cases(
        x_fcast, y_fcast, "trace", y_past[:, -1, None]
    )

    _timeseries(
        x=x_fcast,
        y=y_fcast,
        ax=ax,
        what="fcast",
        color=color_fcast,
        label=f"{forecast_label}",
    )
    ax.set_ylabel(label_y_cum)
    # ax.ylim(ylim_cum)
    prec = 1.0 / (np.log10(ax.get_ylim()[1]) - 2.5)
    if prec < 2.0 and prec >= 0:
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(format_k(int(prec)))
        )

    # --------------------------------------------------------------------------- #
    # Finalize
    # --------------------------------------------------------------------------- #

    for ax in axes:
        ax.set_rasterization_zorder(rcParams.rasterization_zorder)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlim(start, end)
        format_date_xticks(ax)

        # biweekly, remove every second element
        if not axes_provided:
            for label in ax.xaxis.get_ticklabels()[1::2]:
                label.set_visible(False)

    for ax in insets:
        ax.set_xlim(start, model.data_end)
        ax.yaxis.tick_right()
        ax.set_yscale("log")
        if insets_only_two_ticks is True:
            format_date_xticks(ax, minor=False)
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)
            print(ax.xticks)
        else:
            format_date_xticks(ax)
            for label in ax.xaxis.get_ticklabels()[1:-1]:
                label.set_visible(False)

    # legend
    leg_loc = "upper left"
    if draw_insets == True:
        leg_loc = "upper right"
    ax = axes[2]
    ax.legend(loc=leg_loc)
    ax.get_legend().get_frame().set_linewidth(0.0)
    ax.get_legend().get_frame().set_facecolor("#F0F0F0")
    # styling legend elements individually does not work. seems like an mpl bug,
    # changes to fontproperties get applied to all legend elements.
    # for tel in ax.get_legend().get_texts():
    #     if tel.get_text() == "Forecasts:":
    #         # tel.set_fontweight("bold")

    if annotate_watermark:
        add_watermark(axes[1])

    fig.suptitle(
        # using script run time. could use last data point though.
        label_leg_dlim,
        x=0.15,
        y=1.075,
        verticalalignment="top",
        # fontsize="large",
        fontweight="bold",
        # loc="left",
        # horizontalalignment="left",
    )

    # plt.subplots_adjust(wspace=0.4, hspace=0.25)
    if save_to is not None:
        plt.savefig(
            save_to + ".pdf",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )
        plt.savefig(
            save_to + ".png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    # add insets to returned axes. maybe not, general axes style would be applied
    # axes = np.append(axes, insets)

    return fig, axes


def _timeseries(
    x,
    y,
    ax=None,
    what="data",
    draw_ci_95=None,
    draw_ci_75=None,
    draw_ci_50=None,
    date_format=True,
    alpha_ci=None,
    **kwargs,
):
    """
    low-level function to plot anything that has a date on the x-axis.

    Parameters
    ----------
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

    date_format: bool, optional
        Automatic converting of index to dates default:True

    kwargs : dict, optional
        directly passed to plotting mpl.

    Returns
    -------
    ax
    """

    # ------------------------------------------------------------------------------ #
    # Default parameter
    # ------------------------------------------------------------------------------ #

    if draw_ci_95 is None:
        draw_ci_95 = rcParams["draw_ci_95"]

    if draw_ci_75 is None:
        draw_ci_75 = rcParams["draw_ci_75"]

    if draw_ci_50 is None:
        draw_ci_50 = rcParams["draw_ci_50"]

    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 3))

    # still need to fix the last dimension being one
    # if x.shape[0] != y.shape[-1]:
    #     log.exception(f"X rows and y rows do not match: {x.shape[0]} vs {y.shape[0]}")
    #     raise KeyError("Shape mismatch")

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

    if what == "data":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_data"])
        if "marker" not in kwargs:
            kwargs = dict(kwargs, marker="d")
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="None")
    elif what == "fcast":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="--")
    elif what == "model":
        if "color" not in kwargs:
            kwargs = dict(kwargs, color=rcParams["color_model"])
        if "ls" not in kwargs and "linestyle" not in kwargs:
            kwargs = dict(kwargs, ls="-")

    # ------------------------------------------------------------------------------ #
    # plot
    # ------------------------------------------------------------------------------ #
    ax.plot(x, data, **kwargs)

    # overwrite some styles that do not play well with fill_between
    if "linewidth" in kwargs:
        del kwargs["linewidth"]
    if "marker" in kwargs:
        del kwargs["marker"]
    if "label" in kwargs:
        del kwargs["label"]
    kwargs["lw"] = 0
    kwargs["alpha"] = 0.1 if alpha_ci is None else alpha_ci

    if draw_ci_95 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=2.5, axis=0),
            np.percentile(y, q=97.5, axis=0),
            **kwargs,
        )

    if draw_ci_75 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=12.5, axis=0),
            np.percentile(y, q=87.5, axis=0),
            **kwargs,
        )

    del kwargs["alpha"]
    kwargs["alpha"] = 0.2 if alpha_ci is None else alpha_ci

    if draw_ci_50 and y.ndim == 2:
        ax.fill_between(
            x,
            np.percentile(y, q=25.0, axis=0),
            np.percentile(y, q=75.0, axis=0),
            **kwargs,
        )

    # ------------------------------------------------------------------------------ #
    # formatting
    # ------------------------------------------------------------------------------ #
    if date_format:
        format_date_xticks(ax)

    return ax


def _new_cases_to_cum_cases(x, y, what, offset=0):
    """
    so this conversion got ugly really quickly.
    need to check dimensionality of y

    Parameters
    ----------
    x : pandas DatetimeIndex array
        will be padded accordingly

    y : 1d or 2d numpy array
        new cases matching dates in x.
        if 1d, we assume raw data (no samples)
        if 2d, we assume results from trace with 0th dim samples and 1st new cases
        matching x

    what : str
        dirty workaround to differntiate between traces and raw data
        "data" or "trace"

    offset : int or array like
        added to cum sum (should be the known cumulative case number at the
        first date of provided in x)

    Returns
    -------
    x_cum : pandas DatetimeIndex array
        dates of the cumulative cases

    y_cum : nd array
        cumulative cases matching x_cum and the dimension of input y

    Example
    -------
    .. code-block::

        cum_dates, cum_cases = _new_cases_to_cum_cases(new_dates, new_cases)
    """

    # things from the trace have the 0-th dimension for samples. raw data does not
    if what == "trace":
        y_cum = np.cumsum(y, axis=1) + offset
    elif what == "data":
        y_cum = np.nancumsum(y, axis=0) + offset
    else:
        raise ValueError

    # example with offset = 0:
    # y_data new_cases [ 281  451  170 1597]
    # y_data cum_cases [ 281  732  902 2499]

    # so the cumulative used to be one day longer when applying the new cases to the
    # next day, then add a date at the end of the x axis
    # add one element using the existing frequency
    # x_cum = x.union(pd.DatetimeIndex([x[-1] + 1 * x.freq]))
    x_cum = x

    return x_cum, y_cum

