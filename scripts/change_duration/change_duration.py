"""
    # Example: Change duration
    This showcases the effect of sudden changes in effective spreading-rate (or R)
    that can cause a temporary decrease in daily new cases.
    Non-hierarchical model using jhu data (no regions).

    Runtime ~ 1h
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


def plot_daily_cases_and_r_smoothing(keys=None, linestyles=None):
    """
        assumes global variables (dicts):
        * mod
        * tr
        * mod_inf_rep
        * tr_inf_rep
        * mod_inf_sym
        * tr_inf_sym
        * context_clr

    """
    if keys is None:
        keys = ["a", "c"]
    if linestyles is None:
        linestyles = ["-", "--"]

    fig_delay, axes_delay = plt.subplots(4, 1, figsize=(3, 4), constrained_layout=True,)
    fig_r, axes_r = plt.subplots(6, 1, figsize=(3, 6), constrained_layout=True,)

    for key, ls in zip(keys, linestyles):
        trace = tr[key]
        model = mod[key]

        mu = trace["mu"][:, None]
        lambda_t, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "lambda_t"
        )

        # ------------------------------------------------------------------------------ #
        # Figure 1: Delays
        # ------------------------------------------------------------------------------ #

        # R input
        ax = axes_delay[0]
        y = lambda_t[:, :] / mu
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", color=context_clr["other"], ls=ls
        )
        ax.set_title(r"Input: $R = \lambda / \mu$")

        # New infected
        ax = axes_delay[1]
        y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_infected")
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", color=context_clr["infected"], ls=ls
        )
        ax.set_title("new Infected")

        # New symptomatic
        ax = axes_delay[2]
        y, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "new_symptomatic"
        )
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", color=context_clr["symptomatic_2"], ls=ls
        )
        ax.set_title("new Symptomatic")

        # New reported
        ax = axes_delay[3]
        y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_reported")
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", color=context_clr["reported_2"], ls=ls
        )
        ax.set_title("new Reported")

        # ------------------------------------------------------------------------------ #
        # Figure 2: Comparisson for different calculation of R
        # ------------------------------------------------------------------------------ #

        # generation duration (serial interval, roughly the incubation time)
        gd = 4

        # R input
        ax = axes_r[0]
        y = lambda_t[:, :] / mu
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", color=context_clr["other"], ls=ls
        )
        ax.set_title(r"Input: $R = \lambda / \mu$")

        # New symptomatic again
        ax = axes_r[1]
        y, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "new_symptomatic"
        )
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", color=context_clr["symptomatic_2"], ls=ls
        )
        ax.set_title("new Symptomatic")

        # R inferred naive: R_t = Sy_t / Sy_t-gd, gd=4 generation time
        ax = axes_r[2]
        y, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "new_symptomatic"
        )
        # r, x_r = naive_R(y, x, gd=4, match_rki_convention=False)
        # cov19.plot._timeseries(x=x_r, y=r, ax=ax, what="model", color=context_clr["other"], ls=ls)
        r, x_r = naive_R(y, x, gd=4, match_rki_convention=True)
        cov19.plot._timeseries(
            x=x_r, y=r, ax=ax, what="model", color=context_clr["symptomatic"], ls=ls
        )
        ax.set_title(r"$R$ naive")

        # R rki avg 4: on Symptomatics
        ax = axes_r[3]
        y, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "new_symptomatic"
        )
        cov19.plot._timeseries(
            x=x,
            y=RKI_R(y, window=4, gd=gd),
            ax=ax,
            what="model",
            color=context_clr["symptomatic"],
            ls=ls,
        )
        ax.set_title(r"$R$ via RKI method (4 days)")

        # R rki avg 7: on Symptomatics
        ax = axes_r[4]
        y, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "new_symptomatic"
        )
        cov19.plot._timeseries(
            x=x,
            y=RKI_R(y, window=7, gd=gd),
            ax=ax,
            what="model",
            color=context_clr["symptomatic"],
            ls=ls,
        )
        ax.set_title(r"$R$ via RKI method (7 days)")

        # R inferred with our model (1cp) on symptomatic
        ax = axes_r[5]
        trace = tr_inf_sym[key]
        model = mod_inf_sym[key]
        mu = trace["mu"]
        lambda_t, x = cov19.plot._get_array_from_trace_via_date(
            model, trace, "lambda_t"
        )
        y = lambda_t[:, :] / trace["mu"][:, None]
        cov19.plot._timeseries(
            x=x, y=y, ax=ax, what="model", color=context_clr["symptomatic"], ls=ls
        )
        ax.set_title(r"$R$ inferred by SIR model")

        # R inferred with our model (1cp) on reported data
        # ax = axes_r[5]
        # trace = tr_inf_rep[key]
        # model = mod_inf_rep[key]
        # mu = trace["mu"]
        # lambda_t, x = cov19.plot._get_array_from_trace_via_date(
        #     model, trace, "lambda_t"
        # )
        # y = lambda_t[:, :] / trace["mu"][:, None]
        # cov19.plot._timeseries(
        #     x=x,
        #     y=y,
        #     ax=ax,
        #     what="model",
        #     draw_ci_95=True,
        #     color=context_clr["reported"],
        #     ls=ls,
        # )
        # ax.set_title(r"$R$ inferred by SIR model")

    for ax in np.concatenate((axes_delay, axes_r)):
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(labelbottom=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.set_xlim(datetime.datetime(2020, 3, 11), datetime.datetime(2020, 4, 15))
        ax.xaxis.set_major_locator(
            mpl.dates.WeekdayLocator(interval=1, byweekday=mpl.dates.WE)
        )

        if (
            ax.title.get_text() == "new Symptomatic"
            or ax.title.get_text() == "new Infected"
            or ax.title.get_text() == "new Reported"
        ):
            ax.set_ylim(0, None)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_k(int(0))))
        else:
            ax.set_ylim(0.5, 3.5)
            ax.hlines(1, x[0], x[-1], linestyles=":")

    for ax in [axes_r[-1], axes_delay[-1]]:
        ax.axes.get_xaxis().set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.set_xlabel("Time (days)")
        # ax.tick_params(labelbottom=True, labeltop=False)

    # Add letters
    letter_kwargs = dict(x=-0.15, y=1.2, fontweight="bold", size="large")
    axes_r[0].text(s="A", transform=axes_r[0].transAxes, **letter_kwargs)
    axes_r[1].text(s="B", transform=axes_r[1].transAxes, **letter_kwargs)
    axes_r[2].text(s="C", transform=axes_r[2].transAxes, **letter_kwargs)
    axes_r[3].text(s="D", transform=axes_r[3].transAxes, **letter_kwargs)
    axes_r[4].text(s="E", transform=axes_r[4].transAxes, **letter_kwargs)
    axes_r[5].text(s="F", transform=axes_r[5].transAxes, **letter_kwargs)

    axes_delay[0].text(s="A", transform=axes_delay[0].transAxes, **letter_kwargs)
    axes_delay[1].text(s="B", transform=axes_delay[1].transAxes, **letter_kwargs)
    axes_delay[2].text(s="C", transform=axes_delay[2].transAxes, **letter_kwargs)
    axes_delay[3].text(s="D", transform=axes_delay[3].transAxes, **letter_kwargs)

    fig_r.savefig("r_comparison.pdf", **save_kwargs)
    fig_delay.savefig("r_left_right.pdf", **save_kwargs)


def plot_rki_convention(key):

    fig, axes = plt.subplots(2, 1, figsize=(3, 3), constrained_layout=True,)

    # New symptomatic
    ax = axes[0]
    y, x = cov19.plot._get_array_from_trace_via_date(
        mod[key], tr[key], "new_symptomatic"
    )
    cov19.plot._timeseries(
        x=x, y=y, ax=ax, what="model", color=context_clr["symptomatic_2"], ls="-"
    )
    ax.set_title("new Symptomatic")

    # R naive, both go into same plot
    ax = axes[1]
    ax.set_title(r"$R$ methods")
    y, x = cov19.plot._get_array_from_trace_via_date(
        mod[key], tr[key], "new_symptomatic"
    )

    r, x_r = naive_R(y, x, gd=4, match_rki_convention=True)
    cov19.plot._timeseries(
        x=x_r,
        y=r,
        ax=ax,
        what="model",
        color=context_clr["symptomatic_2"],
        ls="-",
        label="right edge (rki)",
    )

    r, x_r = naive_R(y, x, gd=4, match_rki_convention=False)
    cov19.plot._timeseries(
        x=x_r,
        y=r,
        ax=ax,
        what="model",
        color=context_clr["symptomatic"],
        ls="-",
        alpha=0.25,
    )
    cov19.plot._timeseries(
        x=x_r,
        y=r,
        ax=ax,
        what="model",
        color=context_clr["symptomatic"],
        ls="--",
        label="left edge",
    )
    ax.legend(loc="upper right")
    style_legend(ax)

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(labelbottom=False)
        ax.axes.get_xaxis().set_visible(False)
        ax.set_xlim(datetime.datetime(2020, 3, 11), datetime.datetime(2020, 4, 15))
        ax.xaxis.set_major_locator(
            mpl.dates.WeekdayLocator(interval=1, byweekday=mpl.dates.WE)
        )

        if (
            ax.title.get_text() == "new Symptomatic"
            or ax.title.get_text() == "new Infected"
            or ax.title.get_text() == "new Reported"
        ):
            ax.set_ylim(0, None)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(format_k(int(0))))
        else:
            ax.set_ylim(0.5, 3.5)
            ax.hlines(1, x[0], x[-1], linestyles=":")

    for ax in [axes[-1]]:
        ax.axes.get_xaxis().set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.set_xlabel("Time (days)")
        # ax.tick_params(labelbottom=True, labeltop=False)

    letter_kwargs = dict(x=-0.15, y=1.1, fontweight="bold", size="large")
    axes[0].text(s="A", transform=axes[0].transAxes, **letter_kwargs)
    axes[1].text(s="B", transform=axes[1].transAxes, **letter_kwargs)
    fig.savefig("rki_convention.pdf", **save_kwargs)


"""
Global save kwargs for matplotlib.savefig
"""
save_kwargs = dict(transparent=True, format="pdf", dpi=300)


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

    # ------------------------------------------------------------------------------ #
    # From R=3 to R=1.1
    # ------------------------------------------------------------------------------ #
    """
    mu_fixed = 0.13
    lambda_old = 0.39 #R=3
    lambda_new = 0.143 #R=1.1
    """

    # ------------------------------------------------------------------------------ #
    # From R=3 to R=0.9
    # ------------------------------------------------------------------------------ #
    """
    mu_fixed = 0.13
    lambda_old = 0.39 #R=3
    lambda_new = 0.117 #R=0.9
    """

    # ------------------------------------------------------------------------------ #
    # From R=3 to R=1
    # ------------------------------------------------------------------------------ #

    mu_fixed = 0.13
    lambda_old = 0.39  # R=3
    lambda_new = 0.13  # R=0.9

    """
    Create dummy data with fixed parameters and
    run it to obtain a dataset which we later use as new cases obs.
    """
    mod = dict()
    tr = dict()
    for key, duration in zip(["a", "b", "c"], [1, 5, 7]):
        mod[key] = dummy_generator_SIR(
            params_model,
            cp_center=datetime.datetime(2020, 3, 23),
            cp_duration=duration,
            lambda_new=lambda_new,
            lambda_old=lambda_old,
            mu=mu_fixed,
        )
        tr[key] = pm.sample(model=mod[key])

    """
    Use our dummy data and infer with a new SIR model using symptom onset
    """
    tr_inf_sym = dict()
    mod_inf_sym = dict()
    for key in ["a", "b", "c"]:
        mod_inf_sym[key] = create_our_SIR(
            model=mod[key], trace=tr[key], var_for_cases="new_symptomatic", pr_delay=5
        )
        tr_inf_sym[key] = pm.sample(
            model=mod_inf_sym[key], tune=500, draws=500, init="advi+adapt_diag"
        )

    """
    Use our dummy data and infer with a new SIR model using reported data
    """
    tr_inf_rep = dict()
    mod_inf_rep = dict()
    for key in ["a", "b", "c"]:
        mod_inf_rep[key] = create_our_SIR(
            model=mod[key], trace=tr[key], var_for_cases="new_reported", pr_delay=10
        )
        tr_inf_rep[key] = pm.sample(
            model=mod_inf_rep[key], tune=500, draws=500, init="advi+adapt_diag"
        )

    """
        # Plotting
    """

    cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
    cov19.plot.rcParams.draw_ci_50 = False
    cov19.plot.rcParams.draw_ci_75 = False
    cov19.plot.rcParams.draw_ci_95 = False

    context_clr = dict()
    context_clr["reported"] = "darkgreen"
    context_clr["reported_2"] = "forestgreen"
    context_clr["symptomatic"] = "darkred"
    context_clr["symptomatic_2"] = "firebrick"
    context_clr["infected"] = "darkgray"
    context_clr["other"] = "darkblue"

    plot_daily_cases_and_r_smoothing()
    plot_rki_convention(key="a")
