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
import theano
import theano.tensor as tt
import theano.tensor.signal.conv as tt_conv
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append("../../")
    import covid19_inference as cov19

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
lambda_fixed = 0.39


# helper to showcase the reporting delay via lognormal
def delay_lognormal(inp, median, sigma, amplitude=1.0, dist_len=40):
    """ Delays input inp by lognormal distirbution using convolution """
    dist_len = tt.cast(dist_len, "int64")
    beta = cov19.model.utility.tt_lognormal(
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


# we want to create multiple models with the different change points
def create_model(params_model, cp_center, cp_duration, lambda_new):
    with cov19.model.Cov19Model(**params_model) as this_model:

        # usually, make lambda a pymc3 random variable.
        # In this toymodel, we set to a fixed value
        # lambda_old = pm.Lognormal("lambda", mu=0.39, sigma=0.5)
        lambda_old = tt.constant(lambda_fixed)
        pm.Deterministic("lambda", lambda_old)

        # convert date-arguments to (integer) days
        cp_center = (cp_center - this_model.sim_begin).days

        # create a simple time series with the change-point as a smooth step:
        # linear interpolation between 0 and 1
        time_array = np.arange(this_model.sim_len)
        step_t = tt.clip((time_array - cp_center + cp_duration / 2) / cp_duration, 0, 1)

        # create the time series of lambda (on log scale)
        lambda_t = step_t * (lambda_new - lambda_old) + lambda_old
        pm.Deterministic("lambda_t", lambda_t)

        # based on the timeseries of the rates, get new cases (infected) via SIR
        # needs lambda_t on log scale
        new_I_t, new_E_t, I_t, S_t = cov19.model.SEIR(
            lambda_t_log=tt.log(lambda_t),
            mu=mu_fixed,
            # new exposed per day: infected but not infectious yet
            name_new_E_t="new_E_t",
            # new infectious per day
            name_new_I_t="new_I_t",
            # incubation period, lognormal distributed, duration in the E pool
            name_median_incubation="median_incubation",
            return_all =True,
        )
        pm.Deterministic("mu", tt.constant(mu_fixed))

        # SIR our paper
        # delay: S->Reported
        #   * fixed timeshift with lognormal prior
        #   * in ensemble average: lognomal delay
        #   * different from lognormal convolution
        #   * hence, different dynamics

        # SIR now:
        #   * lognorm conv. with (log?)normal prior (kernel) of median
        #   * hence, the delay influences the dynamic within each trace / sample

        # 3 delays: (all pools new per day)
        # Infected -> Infectious        | E->I  | pr_median ~ 4 (in paper 5)
        # Infectious -> Sympotmatic     |       | Rki Refdatum ~ 1 day, not in the code
        # Symptomatic -> Reported       |       | Rki Meldedatum

        # RKI:
        #    Infected -> Infectious  | E         | "Serial interval" 4 days
        #    Infected -> Sympotmatic | E +delay  | "Incubation period" 4-6 days


        # Plots
        # Infected          | E
        # Symptomatic       | E + .. delay ..           | ~ 4-6 day | check cori et al 2013, machtes rki
        # Reported          | I + ^^ large ^^ delay
        #   -> Tests are performed without symptoms, and can be positive when in I pool


        # incubation period
        new_symptomatic = delay_lognormal(inp=new_E_t, median=5, sigma=0.4, dist_len=30)
        pm.Deterministic("new_symptomatic", new_symptomatic)

        # median
        new_reported = delay_lognormal(inp=new_I_t, median=7, sigma=0.4, dist_len=30)
        pm.Deterministic("new_reported", new_reported)


        # here, we do not need to optimize the parameters and can skip the likelihood
        # cov19.model.student_t_likelihood(cases=new_cases)

    return this_model


mod = dict()
mod["a"] = create_model(
    params_model,
    cp_center=datetime.datetime(2020, 3, 23),
    cp_duration=0.1,
    lambda_new=0.15,
)
mod["b"] = create_model(
    params_model,
    cp_center=datetime.datetime(2020, 3, 23),
    cp_duration=4,
    lambda_new=0.15,
)
mod["c"] = create_model(
    params_model,
    cp_center=datetime.datetime(2020, 3, 23),
    cp_duration=8,
    lambda_new=0.15,
)

tr = dict()
tr["a"] = pm.sample(model=mod["a"], tune=50, draws=100)
tr["b"] = pm.sample(model=mod["b"], tune=50, draws=100)
tr["c"] = pm.sample(model=mod["c"], tune=50, draws=100)

"""
    ## Plotting
"""
cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
cov19.plot.rcParams.draw_ci_50 = False
cov19.plot.rcParams.draw_ci_75 = False
cov19.plot.rcParams.draw_ci_95 = False

fig, axes = plt.subplots(
    4,
    1,
    figsize=(6, 8),
    gridspec_kw={"height_ratios": [2, 2, 3, 3]},
    constrained_layout=True,
)

for key, clr in zip(["a", "b", "c"], ["tab:red", "tab:orange", "tab:green"]):
    trace = tr[key]
    model = mod[key]

    mu = trace["mu"][:, None]
    lambda_t, x = cov19.plot._get_array_from_trace_via_date(model, trace, "lambda_t")

    # R
    ax = axes[0]
    y = lambda_t[:, :] / mu
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color=clr)
    ax.set_ylabel(r"$\lambda / \mu$")
    ax.set_ylim(0.8, 3.1)
    if key == "a":
        # only annotate once
        ax.hlines(1, x[0], x[-1], linestyles=":")

    # New infected, not infectious
    ax = axes[1]
    y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_E_t")
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color=clr)
    ax.set_ylabel("New infected\nper day")

    # New symptomatic
    ax = axes[2]
    y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_symptomatic")
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color=clr)
    ax.set_ylabel("New symptomatic\nper day")

    # New reported
    ax = axes[3]
    y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_reported")
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color=clr)
    ax.set_ylabel("New reported\nper day")

    # R inferred naive: R_t = Sy_t / Sy_t-d, d=4 generation time

    # R rki avg 4:

    # R rki avg 7:

    # infer R via our SIR model with 1 cp assumption
    #   * fix prior values for time integration of SEIR generator

    # "if what we inferred (3CP!) was correct, what would the R inferred via RKi etc. look like?!"
    #   * plug in the estimates we inferred in the paper as (fixed) values for the SEIR generation

for ax in axes:
    ax.set_xlim(datetime.datetime(2020, 3, 1), datetime.datetime(2020, 4, 19))
