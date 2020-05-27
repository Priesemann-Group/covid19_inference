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
    sys.path.append("../")
    sys.path.append("../../")
    sys.path.append("../../../")
    import covid19_inference as cov19

def create_our_SIR(model,trace):
    new_cases_obs, time  = cov19.plot._get_array_from_trace_via_date(model,trace,"new_symptomatic")

    diff_data_sim = 16  # should be significantly larger than the expected delay, in
    # order to always fit the same number of data points.
    num_days_forecast = 10

    change_points = [
        dict(
            pr_mean_date_transient=datetime.datetime(2020, 3, 23),
            pr_sigma_date_transient=3,
            pr_median_lambda=0.2,
            pr_sigma_lambda=1)
        ]

    params_model = dict(
        new_cases_obs=new_cases_obs,
        data_begin=time[0],
        fcast_len=num_days_forecast,
        diff_data_sim=16,
        N_population=83e6,
    )

    pr_delay = 5

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
        prior_I = cov19.model.uncorrelated_prior_I(
            lambda_t_log=lambda_t_log,
            mu=mu,
            pr_median_delay=pr_delay,
            name_I_begin="I_begin",
            name_I_begin_ratio_log="I_begin_ratio_log",
            pr_sigma_I_begin=2,
            n_data_points_used=5,
        )

        # Use lambda_t_log and mu to run the SIR model
        new_cases = cov19.model.SIR(
            lambda_t_log=lambda_t_log,
            mu=mu,
            name_new_I_t="new_I_t",
            name_I_t="I_t",
            name_I_begin="I_begin",
            pr_I_begin=prior_I,
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

# helper to showcase the reporting delay via lognormal
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


# we want to create multiple models with the different change points
def create_fixed_model(params_model, cp_center, cp_duration, lambda_new):
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
            pr_mean_median_incubation=4,
            name_median_incubation="median_incubation",
            # fix I_begin, new_E_begin and incubation time instead of inferring them
            # with pymc3
            pr_I_begin=tt.constant(100, dtype="float64"),
            # dirty workaround, we need shape 11 for the convolution running in SEIR
            pr_new_E_begin=tt.ones(11, dtype="float64")*50,
            # another dirty workaround so we keep one free variable but it is alwys the same effectively
            pr_sigma_median_incubation=0.1,
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
        new_symptomatic = delay_lognormal(inp=new_E_t, median=5, sigma=0.3, dist_len=30)
        pm.Deterministic("new_symptomatic", new_symptomatic)

        # the other stuff
        new_reported = delay_lognormal(inp=new_I_t, median=7, sigma=0.3, dist_len=30)
        pm.Deterministic("new_reported", new_reported)


        # here, we do not need to optimize the parameters and can skip the likelihood
        # cov19.model.student_t_likelihood(cases=new_cases)

    return this_model


# ------------------------------------------------------------------------------ #
# Start here
# ------------------------------------------------------------------------------ #
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

"""
Create dummy data with fixed parameters and
run it to obtain a dataset which we later use as new cases obs.
"""
mod = dict()
mod["a"] = create_fixed_model(
    params_model,
    cp_center=datetime.datetime(2020, 3, 23),
    cp_duration=0.1,
    lambda_new=0.15,
)
mod["b"] = create_fixed_model(
    params_model,
    cp_center=datetime.datetime(2020, 3, 23),
    cp_duration=4,
    lambda_new=0.15,
)
mod["c"] = create_fixed_model(
    params_model,
    cp_center=datetime.datetime(2020, 3, 23),
    cp_duration=8,
    lambda_new=0.15,
)

tr = dict()
tr["a"] = pm.sample(model=mod["a"])
tr["b"] = pm.sample(model=mod["b"])
tr["c"] = pm.sample(model=mod["c"])

"""
Use our dummy data to run a new model 
"""
sir_mod = dict()
sir_mod["a"] = create_our_SIR(mod["a"],tr["a"])
sir_mod["b"] = create_our_SIR(mod["b"],tr["b"])
sir_mod["c"] = create_our_SIR(mod["c"],tr["c"])

sir_tr = dict()
sir_tr["a"] = pm.sample(model=sir_mod["a"], tune=50, draws=100,init="advi+adapt_diag")
sir_tr["b"] = pm.sample(model=sir_mod["b"], tune=50, draws=100,init="advi+adapt_diag")
sir_tr["c"] = pm.sample(model=sir_mod["c"], tune=50, draws=100,init="advi+adapt_diag")

ax = cov19.plot.timeseries_overview(sir_mod["a"], sir_tr["a"])
plt.show()

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
    ax.set_ylabel("new Infected")

    # New symptomatic
    ax = axes[2]
    y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_symptomatic")
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color=clr)
    ax.set_ylabel("new Symptomatic")

    # New reported
    ax = axes[3]
    y, x = cov19.plot._get_array_from_trace_via_date(model, trace, "new_reported")
    cov19.plot._timeseries(x=x, y=y, ax=ax, what="model", color=clr)
    ax.set_ylabel("new Reported")

    # R inferred naive: R_t = Sy_t / Sy_t-d, d=4 generation time

    # R rki avg 4:

    # R rki avg 7:

    # infer R via our SIR model with 1 cp assumption
    #   * fix prior values for time integration of SEIR generator

    # "if what we inferred (3CP!) was correct, what would the R inferred via RKi etc. look like?!"
    #   * plug in the estimates we inferred in the paper as (fixed) values for the SEIR generation

for ax in axes:
    ax.set_xlim(datetime.datetime(2020, 3, 1), datetime.datetime(2020, 4, 19))
