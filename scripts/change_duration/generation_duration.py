"""
    This script can be executed to create a figure showing the
    effect of different generation durations on the example of
    the RKI_R method.

    Runtime ~ 1 min
"""

import datetime
import copy
import sys

import pymc3 as pm
import numpy as np
import pandas as pd
import theano
import theano.tensor as tt
import theano.tensor.signal.conv as tt_conv
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append("../")
    sys.path.append("../../")
    sys.path.append("../../../")
    import covid19_inference as cov19

from helper_functions import *


def plot_generation_duration(model, trace, gds, lim_y, axes=None):
    """
        Helper function that creates generation duration plots with RKI_R
        method with a window of 4!

        Parameters
        ----------
        model:
        trace:
            the trace which get used to calculate the R value
        gds: array
            generation durations to plot can be multiple
        lim_y: array of tuples
            y lims for the gd plots (lower,upper)
        Returns
        -------
        fig_r, axes_r : matplotlib figure and axes
    """
    if not isinstance(gds, list):
        raise ValueError("gds has to be a array")
    if not len(gds) == len(lim_y):
        raise ValueError("Not enought y_lims")
    if axes is None:
        fig, axes = plt.subplots(len(gds), 1, figsize=(3, 5))
    if not len(axes) == len(gds):
        raise ValueError("Shape missmatch axes and gds")
    """
    Plot different gd
    """
    for i, gd in enumerate(gds):
        ax = axes[i]
        y, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "new_symptomatic"
        )
        y = RKI_R(y, window=4, gd=gd)
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", color=clr, label=f"gd={gd}"
        )
        ax.legend()
    """
    Format axes
    """
    for i, ax in enumerate(axes):
        ax.set_ylim(lim_y[i][0], lim_y[i][1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(labelbottom=False)
        ax.set_xlim(datetime.datetime(2020, 3, 11), datetime.datetime(2020, 4, 15))
        ax.xaxis.set_major_locator(
            mpl.dates.WeekdayLocator(interval=1, byweekday=mpl.dates.WE)
        )
        ax.hlines(1, x[0], x[-1], linestyles=":")

    axes[0].set_title(r"$R$ via RKI method (4 days)")
    axes[-1].set_xlabel("Time (days)")
    axes[-1].tick_params(labelbottom=True, labeltop=False)
    return axes


"""
Only run if this file is executed helps with import of plotting function defined above,
but cant be run as iypthon notebook
"""
if __name__ == "__main__":
    """
        # Parameterization
    """
    cov19.log.loglevel = "DEBUG"

    # limit the data range to exponential growth
    bd = datetime.datetime(2020, 3, 2)
    ed = datetime.datetime(2020, 3, 15)

    # download data
    # the toy model could do without data but the toolbox assumes some realworld data.
    jhu = cov19.data_retrieval.JHU(auto_download=True)
    cum_cases = jhu.get_total(country="Germany", data_begin=bd, data_end=ed)
    new_cases = jhu.get_new(country="Germany", data_begin=bd, data_end=ed)

    # set model parameters
    params_model = dict(
        new_cases_obs=new_cases,
        data_begin=bd,
        fcast_len=45,
        diff_data_sim=16,
        N_population=83e6,
    )

    # hard coded recovery rate and initial spreading rate, R ~ 3
    mu_fixed = 0.13
    lambda_old = 0.39
    lambda_new = 0.15

    # for SEIR other values are needed
    # mu_fixed = 0.35
    # lambda_old = 1.2
    # lambda_new = 0.40

    """
    Create dummy data with fixed parameters and
    run it to obtain a dataset which we later use as new cases obs.
    """
    model = dummy_generator_SIR(
        params_model,
        cp_center=datetime.datetime(2020, 3, 23),
        cp_duration=1,
        lambda_new=lambda_new,
        lambda_old=lambda_old,
        mu=mu_fixed,
    )
    trace = pm.sample(model=model)

    """
        ## Plotting
    """
    clr = "tab:green"
    cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
    cov19.plot.rcParams.draw_ci_50 = False
    cov19.plot.rcParams.draw_ci_75 = False
    cov19.plot.rcParams.draw_ci_95 = False

    fig, axes = plt.subplots(5, 1, figsize=(5, 6), constrained_layout=True,)

    mu = trace["mu"][:, None]
    lambda_t, x = cov19.plot._get_array_from_trace_via_date(model, trace, "lambda_t")

    def y_fmt(x, y):
        x = x / 1000
        return f"{x:.0f} k"

    # R input
    ax = axes[0]
    y = lambda_t[:, :] / mu
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color=clr)
    ax.set_title(r"Input: $R = \lambda / \mu$")
    cov19.plot._format_date_xticks(ax)

    ax = axes[1]
    y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_symptomatic")
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color=clr)
    ax.set_title("new Symptomatic")
    cov19.plot._format_date_xticks(ax)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(y_fmt))

    # 3 axes for different generation durations
    plot_generation_duration(
        model=model,
        trace=trace,
        axes=axes[2:],
        gds=[4, 7, 11],
        lim_y=[
            (0.1, 3.05),
            (0.1, 7),
            (-0.6, 14.9),
        ],  # generation duration (serial interval, roughly the incubation time)
    )

    # Format x_axis
    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(labelbottom=False)
        ax.set_xlim(datetime.datetime(2020, 3, 11), datetime.datetime(2020, 4, 15))
        ax.xaxis.set_major_locator(
            mpl.dates.WeekdayLocator(interval=1, byweekday=mpl.dates.WE)
        )

    # Label on last axis
    axes[-1].set_xlabel("Time (days)")
    axes[-1].tick_params(labelbottom=True, labeltop=False)

    # Ticks on top first R via RKI method
    # axes[2].tick_params(labelbottom=False, labeltop=True)
    # axes[2].xaxis.tick_top()

    # Label on input R and new Symptomatic
    axes[0].tick_params(labelbottom=True)
    axes[1].tick_params(labelbottom=True)

    axes[-3].set_yticks([1,2,3])
    axes[-2].set_yticks([1,3,5,7])
    axes[-1].set_yticks([1,5,9,13])