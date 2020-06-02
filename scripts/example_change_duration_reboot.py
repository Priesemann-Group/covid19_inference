"""
    # Example: Change duration
    This showcases the effect of sudden changes in effective spreading-rate (or R)
    that can cause a temporary decrease in daily new cases.
    Non-hierarchical model using jhu data (no regions).

    Runtime ~ 3 min
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


"""
# helper functions
"""


def RKI_R(infected_t, window=4, gd=4):
    """ Calculate R value as published by the RKI on 2020-05-15
    in 'Erkäuterung der Schätzung der zeitlich variierenden Reproduktionszahl R'

    infected: Timeseries or trace, in general: last index is considered to calculate R
    window: averaging window, average over 4 days is default
    gd: generation duration, default 4 days. assuming this to be larger increases R
    """
    r = np.zeros(infected_t.shape)  # Empty array with the same shape as Timeseries

    if window == 1:
        r[..., gd:] = infected_t[..., gd:] / infected_t[..., :-gd]
    else:
        if window == 7:
            offset = 1
        elif window == 4:
            offset = 0

        for t in range(window + gd, infected_t.shape[-1] - offset):
            # NOTE: R7_Wert[t-1] <- sum(data$NeuErk[t-0:6]) / sum(data&NeuErk[t-4:10])
            # Indexing in R (the stat-language) is inclusive, in numpy exclusive: upper boundary in sum is increased by 1

            right_block = infected_t[..., t - window : t]
            left_block = infected_t[..., t - window - gd : t - gd]

            r[..., t - 1 - offset] = np.sum(right_block, axis=-1) / np.sum(
                left_block, axis=-1
            )

    # mask of R_values that were not calculated to match time-index of output to input
    return np.ma.masked_where((r == 0) | np.isinf(r) | np.isnan(r), r)


def naive_R(y, x, gd=4, match_rki_convention=False):
    if not match_rki_convention:
        # R naive as in-degree
        r = [y[:, i + gd] / y[:, i] for i in range(0, y.shape[-1] - gd)]
        x = x[:-gd]
    else:
        # R naive as out-degree
        r = [y[:, i] / y[:, i - gd] for i in range(gd, y.shape[-1])]
        x = x[gd:]

    return np.transpose(np.array(r)), x


def create_our_SIR(model, trace, var_for_cases="new_symptomatic", cps=1, pr_delay=5):
    new_cases_obs, time = cov19.plot._get_array_from_trace_via_date(
        model, trace, var_for_cases, start=model.data_begin
    )

    diff_data_sim = model.diff_data_sim  # this should to match the input model
    num_days_forecast = 1  # we do not need a forecast, only inference

    if cps == 1:
        # this matches the step we create in out toy-example
        # the date specifies the middle of the step (passed to lambda_t_with_sigmoids)
        change_points = [
            dict(
                pr_mean_date_transient=datetime.datetime(2020, 3, 23),
                pr_sigma_date_transient=3,
                pr_median_lambda=0.2,
                pr_sigma_lambda=1,
            )
        ]
    elif cps == 3:
        change_points = [
            dict(
                pr_mean_date_transient=datetime.datetime(2020, 3, 9),
                pr_sigma_date_transient=3,
                pr_median_lambda=0.2,
                pr_sigma_lambda=0.5,
            ),
            dict(
                pr_mean_date_transient=datetime.datetime(2020, 3, 16),
                pr_sigma_date_transient=1,
                pr_median_lambda=0.1,
                pr_sigma_lambda=0.5,
            ),
            dict(
                pr_mean_date_transient=datetime.datetime(2020, 3, 23),
                pr_sigma_date_transient=1,
                pr_median_lambda=0.05,
                pr_sigma_lambda=0.5,
            ),
        ]

    params_model = dict(
        new_cases_obs=np.median(new_cases_obs, axis=0),
        data_begin=model.data_begin,
        fcast_len=num_days_forecast,
        diff_data_sim=diff_data_sim,
        N_population=83e6,
    )

    with cov19.model.Cov19Model(**params_model) as this_model:
        # Create the an array of the time dependent infection rate lambda
        lambda_t_log = cov19.model.lambda_t_with_sigmoids(
            pr_median_lambda_0=0.4,
            pr_sigma_lambda_0=0.5,
            change_points_list=change_points,  # The change point priors we constructed earlier
            name_lambda_t="lambda_t",  # Name for the variable in the trace (see later)
        )
        # set prior distribution for the recovery rate
        mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)
        # This builds a decorrelated prior for I_begin for faster inference.
        # It is not necessary to use it, one can simply remove it and use the default argument
        # for pr_I_begin in cov19.SIR

        # Use lambda_t_log and mu to run the SIR model
        new_cases = cov19.model.SIR(
            lambda_t_log=lambda_t_log,
            mu=mu,
            name_new_I_t="new_I_t",
            name_I_t="I_t",
            name_I_begin="I_begin",
        )

        # Delay the cases by a lognormal reporting delay
        new_cases = cov19.model.delay_cases(
            cases=new_cases,
            name_cases="delayed_cases",
            name_delay="delay",
            name_width="delay-width",
            pr_mean_of_median=pr_delay,
            pr_sigma_of_median=0.2,
            pr_median_of_width=0.3,
        )

        # Define the likelihood, uses the new_cases_obs set as model parameter
        cov19.model.student_t_likelihood(new_cases)
    return this_model


# reporting delay via lognormal kernel
def delay_lognormal(inp, median, sigma, amplitude=1.0, dist_len=40):
    """ Delays input inp by lognormal distirbution using convolution """
    dist_len = tt.cast(dist_len, "int64")
    beta = cov19.model._utility.tt_lognormal(
        tt.arange(dist_len * 2), tt.log(median), sigma
    )

    def conv1d(inp, filt, amplitude=1.0):
        """ wraps theano conv2d function for 1d arrays """
        amplitude = tt.cast(amplitude, "float64")
        amplitude = tt.clip(amplitude, 1e-12, 1e9)  # Limit to prevent NANs

        zero = tt.zeros_like(inp)
        a0rp = tt.concatenate((inp, zero,), 0) * amplitude

        a0rp3d = tt.alloc(0.0, 1, a0rp.shape[0], 1)
        a0rp = tt.set_subtensor(a0rp3d[0, :, 0], a0rp)
        filt3d = tt.alloc(0.0, 1, filt.shape[0], 1)
        filt = tt.set_subtensor(filt3d[0, :, 0], filt)
        return tt_conv.conv2d(a0rp, filt, None, None, border_mode="full").flatten()

    return conv1d(inp, beta, amplitude)[: inp.shape[0]]


# hard coded time series of the median lamda values we inferred before
def get_lambda_t_3cp_from_paper():
    # as in Fig 3 https://arxiv.org/abs/2004.01105
    # fmt: off
    y = np.array(
        [0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520853, 0.42520093, 0.42517466, 0.4249035, 0.4236892, 0.41947994, 0.40930175, 0.38876708, 0.34942187, 0.29720015, 0.26635061, 0.25552907, 0.25118942, 0.24956378, 0.24907948, 0.24837864, 0.24296566, 0.22420404, 0.19470871, 0.16873707, 0.15696049, 0.15353644, 0.15277346, 0.15087519, 0.14391068, 0.13167679, 0.11795485, 0.10687387, 0.09951218, 0.09551084, 0.09377538, 0.09302759, 0.09289024, 0.09285077, 0.09284367, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485, 0.09283485,
        ])
    # fmt: on

    x = pd.date_range(start="2020-2-15", periods=len(y)).date

    return y, x


# we want to create multiple models with the differently fast steps
def dummy_generator_SIR(
    params_model, cp_center, cp_duration, lambda_old, lambda_new, mu
):
    with cov19.model.Cov19Model(**params_model) as this_model:

        # usually, make lambda a pymc3 random variable.
        # In this toymodel, we set to a fixed value
        # lambda_old = pm.Lognormal("lambda", mu=0.39, sigma=0.5)
        lambda_old = tt.constant(lambda_old)
        pm.Deterministic("lambda", lambda_old)
        pm.Deterministic("mu", tt.constant(mu))

        # convert date-arguments to (integer) days
        cp_center = (cp_center - this_model.sim_begin).days

        # create a simple time series with the change-point as a smooth step:
        # linear interpolation between 0 and 1
        time_array = np.arange(this_model.sim_len)
        step_t = tt.clip(
            (time_array - cp_center + int(cp_duration / 2)) / cp_duration, 0, 1
        )

        # create the time series of lambda
        lambda_t = step_t * (lambda_new - lambda_old) + lambda_old
        pm.Deterministic("lambda_t", lambda_t)

        # we need at least one random variable for pymc3. use I_0 as a dummy variable
        new_I_t = cov19.model.SIR(
            lambda_t_log=tt.log(lambda_t),
            mu=mu,
            name_new_I_t="new_infected",
            pr_I_begin=10,
        )

        # incubation period
        new_symptomatic = delay_lognormal(inp=new_I_t, median=5, sigma=0.3, dist_len=30)
        pm.Deterministic("new_symptomatic", new_symptomatic)

        # the other stuff
        new_reported = delay_lognormal(inp=new_I_t, median=10, sigma=0.3, dist_len=30)
        pm.Deterministic("new_reported", new_reported)

    return this_model


def plot_distributions(model, trace):
    fig, axes = plt.subplots(6, 3, figsize=(6, 6.4))

    for i, key in enumerate(
        # left column
        ["weekend_factor", "mu", "lambda_0", "lambda_1", "lambda_2", "lambda_3"]
    ):
        try:
            cov19.plot._distribution(model, trace, key, ax=axes[i, 0])
        except:
            print(f"{key} not in vars")

    for i, key in enumerate(
        # mid column
        [
            "offset_modulation",
            "sigma_obs",
            "I_begin",
            "transient_day_1",
            "transient_day_2",
            "transient_day_3",
        ]
    ):
        try:
            cov19.plot._distribution(model, trace, key, ax=axes[i, 1])
        except:
            print(f"{key} not in vars")

    for i, key in enumerate(
        # right column
        ["delay", "transient_len_1", "transient_len_2", "transient_len_3",]
    ):
        try:
            cov19.plot._distribution(model, trace, key, ax=axes[i + 2, 2])
        except:
            print(f"{key} not in vars")

    fig.tight_layout()
    return fig, axes


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
mod = dict()
tr = dict()
for key, duration in zip(["a", "b", "c"], [1, 5, 9]):
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
    ## Plotting
"""

cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
cov19.plot.rcParams.draw_ci_50 = False
cov19.plot.rcParams.draw_ci_75 = False
cov19.plot.rcParams.draw_ci_95 = False

context_clr = dict()
context_clr["reported"] = "darkgreen"
context_clr["symptomatic"] = "darkred"
context_clr["symptomatic_2"] = "firebrick"
context_clr["infected"] = "darkgray"
context_clr["other"] = "darkblue"


def _format_k(prec):
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


def plot_daily_cases_and_r_smoothing(keys=None, linestyles=None):
    """
        assumes global variables (dicts):
        * mod
        * tr
        * mod_inf_rep
        * tr_inf_rep
        * mod_inf_sym
        * tr_inf_sym

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
            x=x, y=y, ax=ax, what="model", color=context_clr["reported"], ls=ls
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
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_format_k(int(0))))
        else:
            ax.set_ylim(0.5, 3.5)
            ax.hlines(1, x[0], x[-1], linestyles=":")

    for ax in [axes_r[-1], axes_delay[-1]]:
        ax.axes.get_xaxis().set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.set_xlabel("Time (days)")
        # ax.tick_params(labelbottom=True, labeltop=False)


plot_daily_cases_and_r_smoothing()
