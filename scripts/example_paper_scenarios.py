# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-05-12 17:09:38
# @Last Modified: 2020-05-13 12:40:42
# ------------------------------------------------------------------------------ #
# Reproduce Dehning et al. arXiv:2004.01105 Figure 3
# ------------------------------------------------------------------------------ #

import datetime

import pymc3 as pm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append("..")
    import covid19_inference as cov19

bd = datetime.datetime(2020, 3, 2)
ed = datetime.datetime(2020, 4, 21)

jhu = cov19.data_retrieval.JHU(auto_download=True)
cum_cases = jhu.get_total(country="Germany", data_begin=bd, data_end=ed)
new_cases = jhu.get_new(country="Germany", data_begin=bd, data_end=ed)

params_model = dict(
    new_cases_obs=new_cases,
    data_begin=bd,
    fcast_len=28,
    diff_data_sim=16,
    N_population=83e6,
)


change_points = [
    # mild distancing
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 3, 9)
        # account for new implementation where transients_day is centered, not begin
        + datetime.timedelta(days=1.5),
        pr_median_transient_len=3,
        pr_sigma_transient_len=0.3,
        pr_sigma_date_transient=3,
        pr_median_lambda=0.2,
        pr_sigma_lambda=0.5,
    ),
    # strong distancing
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 3, 16)
        + datetime.timedelta(days=1.5),
        pr_median_transient_len=3,
        pr_sigma_transient_len=0.3,
        pr_sigma_date_transient=1,
        pr_median_lambda=1 / 8,
        pr_sigma_lambda=0.5,
    ),
    # contact ban
    dict(
        pr_mean_date_transient=datetime.datetime(2020, 3, 23)
        + datetime.timedelta(days=1.5),
        pr_median_transient_len=3,
        pr_sigma_transient_len=0.3,
        pr_sigma_date_transient=1,
        pr_median_lambda=1 / 16,
        pr_sigma_lambda=0.5,
    ),
]

# create a model instance from the parameters and change points from above.
# Add further details.
# Every variable we define in the `with ... as model`-context gets attached
# to the model and becomes a variable in the trace.
with cov19.Cov19Model(**params_model) as model:
    # Create the an array of the time dependent infection rate lambda
    lambda_t_log = cov19.lambda_t_with_sigmoids(
        pr_median_lambda_0=0.4, pr_sigma_lambda_0=0.5, change_points_list=change_points
    )

    # set prior distribution for the recovery rate
    mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)
    pr_median_delay = 8

    # This builds a decorrelated prior for I_begin for faster inference.
    # It is not necessary to use it, one can simply remove it and use the default
    # argument for pr_I_begin in cov19.SIR
    prior_I = cov19.make_prior_I(lambda_t_log, mu, pr_median_delay=pr_median_delay)

    # Use lambda_t_log and mu to run the SIR model
    new_I_t = cov19.SIR(lambda_t_log, mu, pr_I_begin=prior_I)

    # Delay the cases by a lognormal reporting delay
    new_cases_inferred_raw = cov19.delay_cases(
        new_I_t, pr_median_delay=pr_median_delay, pr_median_scale_delay=0.001
    )

    # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend
    # effects
    new_cases_inferred = cov19.week_modulation(new_cases_inferred_raw)

    # Define the likelihood, uses the new_cases_obs set as model parameter
    cov19.student_t_likelihood(new_cases_inferred)


# engage!
trace = pm.sample(model=model, tune=5000, draws=1000, init="advi+adapt_diag")


# ------------------------------------------------------------------------------ #
# plotting
# ------------------------------------------------------------------------------ #

fig, axes = cov19.plot.timeseries_overview(model, trace, offset=cum_cases[0])

fig, axes = plt.subplots(6, 3, figsize=(4, 6.4))
axes[0, 2].set_visible(False)
axes[1, 2].set_visible(False)

# left column
for i, key in enumerate(
    ["weekend_factor", "mu", "lambda_0", "lambda_1", "lambda_2", "lambda_3"]
):
    cov19.plot._distribution(model, trace, key, ax=axes[i, 0])

# mid column
for i, key in enumerate(
    [
        "offset_modulation",
        "sigma_obs",
        "I_begin",
        # beware, these guys were the begin of the transient in the paper,
        # now they are the center points (shifted by transient_len_i)
        "transient_day_1",
        "transient_day_2",
        "transient_day_3",
    ]
):
    cov19.plot._distribution(model, trace, key, ax=axes[i, 1])

# right column
for i, key in enumerate(
    ["delay", "transient_len_1", "transient_len_2", "transient_len_3",]
):
    cov19.plot._distribution(model, trace, key, ax=axes[i + 2, 2])

fig.tight_layout()
