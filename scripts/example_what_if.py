# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-05-14 11:14:19
# @Last Modified: 2020-05-19 10:05:46
# ------------------------------------------------------------------------------ #
# What if scenarios for relaxations around May 11.
# This is a simple example how to construct incorporate expected change points
# (that are not constrained by data yet) to create different szenarios.
# The script also shows a bit how to set the rcParameters for the plots.
# ------------------------------------------------------------------------------ #

import datetime
import copy

import pymc3 as pm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append("..")
    import covid19_inference as cov19

# limit the data range
bd = datetime.datetime(2020, 3, 2)
ed = datetime.datetime(2020, 5, 14)

# download data
jhu = cov19.data_retrieval.JHU(auto_download=True)
cum_cases = jhu.get_total(country="Germany", data_begin=bd, data_end=ed)
new_cases = jhu.get_new(country="Germany", data_begin=bd, data_end=ed)

# set model parameters
params_model = dict(
    new_cases_obs=new_cases,
    data_begin=bd,
    fcast_len=28,
    diff_data_sim=16,
    N_population=83e6,
)


# change points like in the paper
cp_base = [
    # mild distancing
    dict(
        # account for new implementation where transients_day is centered, not begin
        pr_mean_date_transient=datetime.datetime(2020, 3, 10),
        pr_median_transient_len=3,
        pr_sigma_transient_len=0.3,
        pr_sigma_date_transient=3,
        pr_median_lambda=0.2,
        pr_sigma_lambda=0.5,
    ),
    # strong distancing
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 3, 17),
        pr_median_transient_len=3,
        pr_sigma_transient_len=0.3,
        pr_sigma_date_transient=1,
        pr_median_lambda=1 / 8,
        pr_sigma_lambda=0.5,
    ),
    # contact ban
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 3, 24),
        pr_median_transient_len=3,
        pr_sigma_transient_len=0.3,
        pr_sigma_date_transient=1,
        pr_median_lambda=1 / 16,
        pr_sigma_lambda=0.5,
    ),
]

# Scenarios for May 11, due to ~11 days delay, not evident in data yet
# Add additional change points with reference to the previous values,
# we use the posterior value that we inferred before (as of May 14)
ref = 0.10

# a: double the contacts (this only effectively applies apart from family)
cp_a = copy.deepcopy(cp_base)
cp_a.append(
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 5, 12),
        pr_sigma_date_transient=1,
        pr_median_lambda=ref * 2,
        pr_sigma_lambda=0.3,
    ),
)

# b: back to pre-lockdown value
cp_b = copy.deepcopy(cp_base)
cp_b.append(
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 5, 12),
        pr_sigma_date_transient=1,
        pr_median_lambda=0.15,
        pr_sigma_lambda=0.3,
    ),
)

# c: 20% decrease, ideal case, for instance if contact tracing is very effective
cp_c = copy.deepcopy(cp_base)
cp_c.append(
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 5, 12),
        pr_sigma_date_transient=1,
        pr_median_lambda=ref - ref * 0.2,
        pr_sigma_lambda=0.3,
    ),
)


def create_model(change_points, params_model):
    with cov19.Cov19Model(**params_model) as model:
        lambda_t_log = cov19.lambda_t_with_sigmoids(
            pr_median_lambda_0=0.4,
            pr_sigma_lambda_0=0.5,
            change_points_list=change_points,
        )

        mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)
        pr_median_delay = 10

        prior_I = cov19.make_prior_I(lambda_t_log, mu, pr_median_delay=pr_median_delay)

        new_I_t = cov19.SIR(lambda_t_log, mu, pr_I_begin=prior_I)

        new_cases_inferred_raw = cov19.delay_cases(
            new_I_t, pr_median_delay=pr_median_delay, pr_median_scale_delay=0.3
        )

        new_cases_inferred = cov19.week_modulation(new_cases_inferred_raw)

        cov19.student_t_likelihood(new_cases_inferred)

    return model


mod_a = create_model(cp_a, params_model)
mod_b = create_model(cp_b, params_model)
mod_c = create_model(cp_c, params_model)

# engage!
tr_a = pm.sample(model=mod_a, tune=50, draws=10, init="advi+adapt_diag")
tr_b = pm.sample(model=mod_b, tune=50, draws=10, init="advi+adapt_diag")
tr_c = pm.sample(model=mod_c, tune=50, draws=10, init="advi+adapt_diag")
tr_b = tr_a
tr_c = tr_a


# ------------------------------------------------------------------------------ #
# plotting
# ------------------------------------------------------------------------------ #

# english
cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
cov19.plot.rcParams.draw_ci_50 = True

fig, axes = cov19.plot.timeseries_overview(
    mod_a,
    tr_a,
    offset=cum_cases[0],
    forecast_label="Pessimistic",
    forecast_heading=r"$\bf Scenarios\!:$",
    add_more_later=True,
    color="tab:red",
)


fig, axes = cov19.plot.timeseries_overview(
    mod_b,
    tr_b,
    axes=axes,
    offset=cum_cases[0],
    forecast_label="Neutral",
    color="tab:orange",
)

fig, axes = cov19.plot.timeseries_overview(
    mod_c,
    tr_c,
    axes=axes,
    offset=cum_cases[0],
    forecast_label="Optimistic",
    color="tab:green",
)

# german
cov19.plot.set_rcparams(cov19.plot.get_rcparams_default())
cov19.plot.rcParams.draw_ci_50 = True
cov19.plot.rcParams.locale = "de_DE"
cov19.plot.rcParams.date_format = "%-d. %b"

fig, axes = cov19.plot.timeseries_overview(
    mod_a,
    tr_a,
    offset=cum_cases[0],
    forecast_label="Pessimistisch",
    forecast_heading=r"$\bf Szenarien\!:$",
    add_more_later=True,
    color="tab:red",
)


fig, axes = cov19.plot.timeseries_overview(
    mod_b,
    tr_b,
    axes=axes,
    offset=cum_cases[0],
    forecast_label="Neutral",
    color="tab:orange",
)

fig, axes = cov19.plot.timeseries_overview(
    mod_c,
    tr_c,
    axes=axes,
    offset=cum_cases[0],
    forecast_label="Optimistisch",
    color="tab:green",
)
