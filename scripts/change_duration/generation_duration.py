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

#For insets
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_generation_duration(model, trace, gds, lim_y, axes=None, **kwargs):
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
    insets = []
    for i, gd in enumerate(gds):
        ax = axes[i]
        y, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "new_symptomatic"
        )
        y = RKI_R(y, window=4, gd=gd)
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", label=f"$g$={gd}", **kwargs
        )
        ax.hlines(1, x[0], x[-1], linestyles=":")
        ax.set_ylim(lim_y[i][0], lim_y[i][1])
        ax.legend(loc="center left")
        style_legend(ax)

        #Insets
        axin = inset_axes(ax, width="35%", height="50%", loc="upper right")
        cov19.plot._timeseries(x=x,y=y,ax=axin,what="model", **kwargs)
        mark_inset(ax, axin, loc1=2, loc2=4, fc="None", ec="gray", lw=1, alpha=1, zorder=15)
        insets.append(axin)

    """
    Format insets
    """
    x_lims_insets = [
        (datetime.datetime(2020, 3, 29),datetime.datetime(2020, 4, 6)),
        (datetime.datetime(2020, 3, 29),datetime.datetime(2020, 4, 9)),
        (datetime.datetime(2020, 4, 2),datetime.datetime(2020, 4, 12))
        ]
    y_lims_insets = [
        (0.75, 1.1),
        (0.75, 1.1),
        (0.75, 1.1),
    ]
    for i, ax in enumerate(insets):
        ax.tick_params(color='gray',bottom=False,labelbottom=False,labelsize=8,left=False,right=True,labelleft=False,labelright=True)
        ax.set_ylim(y_lims_insets[i][0],y_lims_insets[i][1])
        ax.set_xlim(x_lims_insets[i][0], x_lims_insets[i][1])
        ax.hlines(1, x[0], x[-1], linestyles=":")
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
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
        # Plotting
    """

    cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
    cov19.plot.rcParams.draw_ci_50 = False
    cov19.plot.rcParams.draw_ci_75 = False
    cov19.plot.rcParams.draw_ci_95 = False

    fig, axes = plt.subplots(5, 1, figsize=(4, 5), constrained_layout=False,)

    mu = trace["mu"][:, None]
    lambda_t, x = cov19.plot._get_array_from_trace_via_date(model, trace, "lambda_t")

    # R input
    ax = axes[0]
    y = lambda_t[:, :] / mu
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color="darkblue")
    ax.set_title(r"Input: $R = \lambda / \mu$")
    ax.hlines(1, x[0], x[-1], linestyles=":")
    cov19.plot._format_date_xticks(ax)
    ax.set_ylim(0.5, 3.5)

    # cases symptomatic
    ax = axes[1]
    y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_symptomatic")
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color="firebrick")
    ax.set_title("new Symptomatic")
    cov19.plot._format_date_xticks(ax)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_k(0)))

    # 3 axes for different generation durations
    plot_generation_duration(
        model=model,
        trace=trace,
        axes=axes[2:],
        gds=[2, 4, 8],
        lim_y=[
            (0.5, 1.755),
            (0.1, 3.05),
            (-0.6, 7.05),
        ],  # generation duration (serial interval, roughly the incubation time)
        color="darkred",
    )

    # Format x_axis
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.tick_params(labelbottom=False)
        ax.set_xlim(datetime.datetime(2020, 3, 18), datetime.datetime(2020, 4, 15))
        ax.xaxis.set_major_locator(
            mpl.dates.WeekdayLocator(interval=1, byweekday=mpl.dates.WE)
        )

    # Label on last axis
    axes[-1].spines["bottom"].set_visible(True)
    axes[-1].axes.get_xaxis().set_visible(True)
    axes[-1].set_xlabel("Time (days)")
    # axes[-1].tick_params(labelbottom=True, labeltop=False)

    axes[2].set_title("R via RKI method (4 day average)")

    axes[-3].set_yticks([1.0, 1.5])
    axes[-2].set_yticks([1.0, 2.0, 3.0])
    axes[-1].set_yticks([1.0, 3.0, 5.0, 7.0])
    axes[1].set_yticks([0,15000])
    #Add letters
    letter_kwargs = dict(x=-0.15, y=1.2, fontweight='bold',size="large")
    axes[0].text(s="A", transform=axes[0].transAxes, **letter_kwargs)
    axes[1].text(s="B", transform=axes[1].transAxes, **letter_kwargs)
    axes[2].text(s="C", transform=axes[2].transAxes, **letter_kwargs)
    axes[3].text(s="D", transform=axes[3].transAxes, **letter_kwargs)
    axes[4].text(s="E", transform=axes[4].transAxes, **letter_kwargs)
    fig.tight_layout()

    plt.subplots_adjust(top=0.915,
    bottom=0.084,
    left=0.149,
    right=0.897,
    hspace=1.0,
    wspace=0.2)
    plt.savefig(
    "r_different_generations.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05,
    )
