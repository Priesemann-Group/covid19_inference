# ------------------------------------------------------------------------------ #
# Implementations of the SIR and SEIR-like models
# ------------------------------------------------------------------------------ #

import logging
import numpy as np
import pymc as pm

from aesara import scan
import aesara.tensor as at

from .model import *
from . import utility as ut

log = logging.getLogger(__name__)


def SIR(
    lambda_t_log,
    mu,
    name_new_I_t="new_I_t",
    name_I_begin="I_begin",
    name_I_t="I_t",
    name_S_t="S_t",
    pr_I_begin=100,
    model=None,
    return_all=False,
):
    r"""
    Implements the susceptible-infected-recovered model.

    Parameters
    ----------
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
        time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional the first
        dimension is time.
    mu : :class:`~aesara.tensor.TensorVariable`
        the recovery rate :math:`\mu`, typically a random variable. Can be 0 or 1-dimensional. If 1-dimensional,
        the dimension are the different regions.
    name_new_I_t : str, optional
        Name of the ``new_I_t`` variable
    name_I_begin : str, optional
        Name of the ``I_be  gin`` variable
    name_I_t : str, optional
        Name of the ``I_t`` variable, set to None to avoid adding as trace variable.
    name_S_t : str, optional
        Name of the ``S_t`` variable, set to None to avoid adding as trace variable.
    pr_I_begin : float or array_like or :class:`~aesara.tensor.Variable`
        Prior beta of the Half-Cauchy distribution of :math:`I(0)`.
        if type is ``at.Constant``, I_begin will not be inferred by pymc3
    model : :class:`Cov19Model`
        if none, it is retrieved from the context
    return_all : bool
        if True, returns ``name_new_I_t``, ``name_I_t``, ``name_S_t`` otherwise returns only ``name_new_I_t``

    Returns
    ------------------
    new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.
    I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the infected (if return_all set to True)
    S_t : :class:`~aesara.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)

    """
    log.info("SIR")
    model = modelcontext(model)

    # Total number of people in population
    N = model.N_population

    # Prior distributions of starting populations (infectious, susceptibles)
    if isinstance(pr_I_begin, at.Variable):
        I_begin = pr_I_begin
    else:
        I_begin = pm.HalfCauchy(
            name=name_I_begin, beta=pr_I_begin, shape=model.shape_of_regions
        )

    S_begin = N - I_begin

    lambda_t = at.exp(lambda_t_log)
    new_I_0 = at.zeros_like(I_begin)

    # Runs SIR model:
    def next_day(lambda_t, S_t, I_t, _, mu, N):
        new_I_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - mu * I_t
        I_t = at.clip(I_t, -1, N)  # for stability
        S_t = at.clip(S_t, 0, N)
        return S_t, I_t, new_I_t

    # aesara scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, I, new_I
    outputs, _ = scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[S_begin, I_begin, new_I_0],
        non_sequences=[mu, N],
    )
    S_t, I_t, new_I_t = outputs
    pm.Deterministic(name_new_I_t, new_I_t)
    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_I_t is not None:
        pm.Deterministic(name_I_t, I_t)

    if return_all:
        return new_I_t, I_t, S_t
    else:
        return new_I_t


def SEIR(
    lambda_t_log,
    mu,
    name_new_I_t="new_I_t",
    name_new_E_t="new_E_t",
    name_I_t="I_t",
    name_S_t="S_t",
    name_I_begin="I_begin",
    name_new_E_begin="new_E_begin",
    name_median_incubation="median_incubation",
    pr_I_begin=100,
    pr_new_E_begin=50,
    pr_median_mu=1 / 8,
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    pr_sigma_mu=0.2,
    model=None,
    return_all=False,
):
    r"""
    Implements a model similar to the susceptible-exposed-infected-recovered model.
    Instead of a exponential decaying incubation period, the length of the period is
    lognormal distributed.

    Parameters
    ----------
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
        time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional, the first
        dimension is time.

    mu : :class:`~aesara.tensor.TensorVariable`
        the recovery rate :math:`\mu`, typically a random variable. Can be 0 or
        1-dimensional. If 1-dimensional, the dimension are the different regions.

    name_new_I_t : str, optional
        Name of the ``new_I_t`` variable

    name_I_t : str, optional
        Name of the ``I_t`` variable

    name_S_t : str, optional
        Name of the ``S_t`` variable

    name_I_begin : str, optional
        Name of the ``I_begin`` variable

    name_new_E_begin : str, optional
        Name of the ``new_E_begin`` variable

    name_median_incubation : str
        The name under which the median incubation time is saved in the trace

    pr_I_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`I(0)`.
        if type is ``at.Variable``, ``I_begin`` will be set to the provided prior as
        a constant.

    pr_new_E_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`E(0)`.

    pr_median_mu : float or array_like
        Prior for the median of the
        :class:`~pymc3.distributions.continuous.Lognormal` distribution of the
        recovery rate :math:`\mu`.

    pr_mean_median_incubation :
        Prior mean of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        Defaults to 4 days [Nishiura2020]_, which is the median serial interval (the
        important measure here is not exactly the incubation period, but the delay
        until a person becomes infectious which seems to be about 1 day earlier as
        showing symptoms).

    pr_sigma_median_incubation : number or None
        Prior sigma of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        If None, the incubation time will be fixed to the value of
        ``pr_mean_median_incubation`` instead of a random variable
        Default is 1 day.

    sigma_incubation :
        Scale parameter of the :class:`~pymc3.distributions.continuous.Lognormal`
        distribution of the incubation time/ delay until infectiousness. The default
        is set to 0.4, which is about the scale found in [Nishiura2020]_,
        [Lauer2020]_.

    pr_sigma_mu : float or array_like
        Prior for the sigma of the lognormal distribution of recovery rate
        :math:`\mu`.

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    return_all : bool
        if True, returns ``name_new_I_t``, ``name_new_E_t``,  ``name_I_t``,
        ``name_S_t`` otherwise returns only ``name_new_I_t``

    Returns
    -------
    name_new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.

    name_new_E_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly exposed persons. (if return_all set to
        True)

    name_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the infected (if return_all set to True)

    name_S_t : :class:`~aesara.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)

    """
    log.info("SEIR")
    model = modelcontext(model)

    # Build prior distrubutions:
    # --------------------------

    # Total number of people in population
    N = model.N_population

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if isinstance(pr_new_E_begin, at.Variable):
        new_E_begin = pr_new_E_begin
    else:
        if not model.is_hierarchical:
            new_E_begin = pm.HalfCauchy(
                name=name_new_E_begin, beta=pr_new_E_begin, shape=11
            )
        else:
            new_E_begin = pm.HalfCauchy(
                name=name_new_E_begin,
                beta=pr_new_E_begin,
                shape=(11, model.shape_of_regions),
            )

    # Prior distributions of starting populations (infectious, susceptibles)
    if isinstance(pr_I_begin, at.Variable):
        I_begin = pr_I_begin
    else:
        I_begin = pm.HalfCauchy(
            name=name_I_begin, beta=pr_I_begin, shape=model.shape_of_regions
        )

    S_begin = N - I_begin - pm.math.sum(new_E_begin, axis=0)

    lambda_t = at.exp(lambda_t_log)
    new_I_0 = at.zeros_like(I_begin)

    if pr_sigma_median_incubation is None:
        median_incubation = pr_mean_median_incubation
    else:
        median_incubation = pm.Normal(
            name_median_incubation,
            mu=pr_mean_median_incubation,
            sigma=pr_sigma_median_incubation,
        )

    # Choose transition rates (E to I) according to incubation period distribution
    if not model.is_hierarchical:
        x = np.arange(1, 11)
    else:
        x = np.arange(1, 11)[:, None]

    beta = ut.tt_lognormal(x, at.log(median_incubation), sigma_incubation)

    # Runs SEIR model:
    def next_day(
        lambda_t,
        S_t,
        nE1,
        nE2,
        nE3,
        nE4,
        nE5,
        nE6,
        nE7,
        nE8,
        nE9,
        nE10,
        I_t,
        _,
        mu,
        beta,
        N,
    ):
        new_E_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_E_t
        new_I_t = (
            beta[0] * nE1
            + beta[1] * nE2
            + beta[2] * nE3
            + beta[3] * nE4
            + beta[4] * nE5
            + beta[5] * nE6
            + beta[6] * nE7
            + beta[7] * nE8
            + beta[8] * nE9
            + beta[9] * nE10
        )
        I_t = I_t + new_I_t - mu * I_t
        I_t = at.clip(I_t, -1, N - 1)  # for stability
        S_t = at.clip(S_t, -1, N)
        return S_t, new_E_t, I_t, new_I_t

    # aesara scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, E's, I, new_I
    outputs, _ = scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[
            S_begin,
            dict(initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]),
            I_begin,
            new_I_0,
        ],
        non_sequences=[mu, beta, N],
    )
    S_t, new_E_t, I_t, new_I_t = outputs
    pm.Deterministic(name_new_I_t, new_I_t)

    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_I_t is not None:
        pm.Deterministic(name_I_t, I_t)
    if name_new_E_t is not None:
        pm.Deterministic(name_new_E_t, new_E_t)

    if return_all:
        return new_I_t, new_E_t, I_t, S_t
    else:
        return new_I_t


def kernelized_spread(
    lambda_t_log,
    name_new_I_t="new_I_t",
    name_new_E_t="new_E_t",
    name_S_t="S_t",
    name_new_E_begin="new_E_begin",
    name_median_incubation="median_incubation",
    pr_new_E_begin=50,
    pr_median_mu=1 / 8,
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    pr_sigma_mu=0.2,
    model=None,
    return_all=False,
):
    r"""
    Implements a model similar to the susceptible-exposed-infected-recovered model.
    Instead of a exponential decaying incubation period, the length of the period is
    lognormal distributed.

    Parameters
    ----------
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
        time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional, the first
        dimension is time.

    mu : :class:`~aesara.tensor.TensorVariable`
        the recovery rate :math:`\mu`, typically a random variable. Can be 0 or
        1-dimensional. If 1-dimensional, the dimension are the different regions.

    name_new_I_t : str, optional
        Name of the ``new_I_t`` variable

    name_S_t : str, optional
        Name of the ``S_t`` variable

    name_new_E_begin : str, optional
        Name of the ``new_E_begin`` variable

    name_median_incubation : str
        The name under which the median incubation time is saved in the trace

    pr_I_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`I(0)`.
        if type is ``at.Variable``, ``I_begin`` will be set to the provided prior as
        a constant.

    pr_new_E_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`E(0)`.

    pr_median_mu : float or array_like
        Prior for the median of the
        :class:`~pymc3.distributions.continuous.Lognormal` distribution of the
        recovery rate :math:`\mu`.

    pr_mean_median_incubation :
        Prior mean of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        Defaults to 4 days [Nishiura2020]_, which is the median serial interval (the
        important measure here is not exactly the incubation period, but the delay
        until a person becomes infectious which seems to be about 1 day earlier as
        showing symptoms).

    pr_sigma_median_incubation : number or None
        Prior sigma of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        If None, the incubation time will be fixed to the value of
        ``pr_mean_median_incubation`` instead of a random variable
        Default is 1 day.

    sigma_incubation :
        Scale parameter of the :class:`~pymc3.distributions.continuous.Lognormal`
        distribution of the incubation time/ delay until infectiousness. The default
        is set to 0.4, which is about the scale found in [Nishiura2020]_,
        [Lauer2020]_.

    pr_sigma_mu : float or array_like
        Prior for the sigma of the lognormal distribution of recovery rate
        :math:`\mu`.

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    return_all : bool
        if True, returns ``name_new_I_t``, ``name_new_E_t``,  ``name_I_t``,
        ``name_S_t`` otherwise returns only ``name_new_I_t``

    Returns
    -------
    name_new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.

    name_new_E_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly exposed persons. (if return_all set to
        True)

    name_S_t : :class:`~aesara.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)

    """
    log.info("kernelized spread")
    model = modelcontext(model)

    # Build prior distrubutions:
    # --------------------------

    # Total number of people in population
    N = model.N_population

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if isinstance(pr_new_E_begin, at.Variable):
        new_E_begin = pr_new_E_begin
    else:
        if not model.is_hierarchical:
            new_E_begin = pm.HalfCauchy(
                name=name_new_E_begin, beta=pr_new_E_begin, shape=11
            )
        else:
            new_E_begin = pm.HalfCauchy(
                name=name_new_E_begin,
                beta=pr_new_E_begin,
                shape=(11, model.shape_of_regions),
            )

    S_begin = N - pm.math.sum(new_E_begin, axis=0)

    lambda_t = at.exp(lambda_t_log)
    new_I_0 = at.zeros(model.shape_of_regions)

    if pr_sigma_median_incubation is None:
        median_incubation = pr_mean_median_incubation
    else:
        median_incubation = pm.Normal(
            name_median_incubation,
            mu=pr_mean_median_incubation,
            sigma=pr_sigma_median_incubation,
        )

    # Choose transition rates (E to I) according to incubation period distribution
    if not model.is_hierarchical:
        x = np.arange(1, 11)
    else:
        x = np.arange(1, 11)[:, None]

    beta = ut.tt_lognormal(x, at.log(median_incubation), sigma_incubation)

    # Runs kernelized spread model:
    def next_day(
        lambda_t,
        S_t,
        nE1,
        nE2,
        nE3,
        nE4,
        nE5,
        nE6,
        nE7,
        nE8,
        nE9,
        nE10,
        _,
        beta,
        N,
    ):
        new_I_t = (
            beta[0] * nE1
            + beta[1] * nE2
            + beta[2] * nE3
            + beta[3] * nE4
            + beta[4] * nE5
            + beta[5] * nE6
            + beta[6] * nE7
            + beta[7] * nE8
            + beta[8] * nE9
            + beta[9] * nE10
        )
        new_E_t = lambda_t / N * new_I_t * S_t
        S_t = S_t - new_E_t
        S_t = at.clip(S_t, -1, N)
        return S_t, new_E_t, new_I_t

    # aesara scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, E's, new_I
    outputs, _ = scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[
            S_begin,
            dict(initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]),
            new_I_0,
        ],
        non_sequences=[beta, N],
    )
    S_t, new_E_t, new_I_t = outputs
    pm.Deterministic(name_new_I_t, new_I_t)

    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_new_E_t is not None:
        pm.Deterministic(name_new_E_t, new_E_t)

    if return_all:
        return new_I_t, new_E_t, S_t
    else:
        return new_I_t


def uncorrelated_prior_I(
    lambda_t_log,
    mu,
    pr_median_delay,
    name_I_begin="I_begin",
    name_I_begin_ratio_log="I_begin_ratio_log",
    pr_sigma_I_begin=2,
    n_data_points_used=5,
    model=None,
):
    r"""
    Builds the prior for I begin  by solving the SIR differential from the first
    data backwards. This decorrelates the I_begin from the lambda_t at the
    beginning, allowing a more efficient sampling. The example_one_bundesland runs
    about 30\% faster with this prior, instead of a HalfCauchy.

    Parameters
    ----------
    lambda_t_log : TYPE
        Description
    mu : TYPE
        Description
    pr_median_delay : TYPE
        Description
    name_I_begin : str, optional
        Description
    name_I_begin_ratio_log : str, optional
        Description
    pr_sigma_I_begin : int, optional
        Description
    n_data_points_used : int, optional
        Description
    model : :class:`Cov19Model`
        if none, it is retrieved from the context
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
    mu : :class:`~aesara.tensor.TensorVariable`
    pr_median_delay : float
    pr_sigma_I_begin : float
    n_data_points_used : int

    Returns
    ------------------
    I_begin: :class:`~aesara.tensor.TensorVariable`

    """
    log.info("Uncorrelated prior_I")
    model = modelcontext(model)

    num_regions = () if model.sim_ndim == 1 else model.sim_shape[1]

    lambda_t = at.exp(lambda_t_log)

    delay = round(pr_median_delay)
    num_new_I_ref = (
        np.nansum(model.new_cases_obs[:n_data_points_used], axis=0) / model.data_len
    )
    if model.is_hierarchical:
        I0_ref = num_new_I_ref / lambda_t[0]
        diff_I_begin_L2_log, diff_I_begin_L1_log = ut.hierarchical_normal(
            name_L1=f"{name_I_begin_ratio_log}_L1",
            name_L2=f"{name_I_begin_ratio_log}_L2",
            name_sigma=f"sigma_{name_I_begin_ratio_log}_L1",
            pr_mean=0,
            pr_sigma=pr_sigma_I_begin // 2,
            error_cauchy=False,
        )
        I_begin = I0_ref * at.exp(diff_I_begin_L2_log)
    else:
        days_diff = model.diff_data_sim - delay + 3
        I_ref = num_new_I_ref / lambda_t[days_diff]
        I0_ref = I_ref / (1 + lambda_t[days_diff // 2] - mu) ** days_diff
        I_begin = I0_ref * at.exp(
            pm.Normal(
                name=name_I_begin_ratio_log,
                mu=0,
                sigma=pr_sigma_I_begin,
                shape=num_regions,
            )
        )
    # diff_I_begin_L2_log, diff_I_begin_L1_log = ut.hierarchical_normal(
    #    name_L1=f"{name_I_begin_ratio_log}_L1",
    #    name_L2=f"{name_I_begin_ratio_log}_L2",
    #    name_sigma=f"sigma_{name_I_begin_ratio_log}_L1",
    #    pr_mean=0,
    #    pr_sigma=pr_sigma_I_begin // 2,
    #    error_cauchy=False,
    # )

    # I_begin = I0_ref * at.exp(diff_I_begin_L2_log)

    # I_begin = pm.Lognormal(
    #   name_I_begin_ratio_log, mu=at.log(I0_ref), sigma=2.5, shape=num_regions
    # )
    pm.Deterministic(name_I_begin, I_begin)
    return I_begin


def uncorrelated_prior_E(
    name_E_begin="I_begin",
    name_E_begin_ratio_log="I_begin_ratio_log",
    pr_sigma_E_begin=2,
    n_data_points_used=5,
    model=None,
    len_time=11,
):
    r"""
    Builds the prior for I begin  by solving the SIR differential from the first
    data backwards. This decorrelates the I_begin from the lambda_t at the
    beginning, allowing a more efficient sampling. The example_one_bundesland runs
    about 30\% faster with this prior, instead of a HalfCauchy.

    Parameters
    ----------
    lambda_t_log : TYPE
        Description
    mu : TYPE
        Description
    pr_median_delay : TYPE
        Description
    name_I_begin : str, optional
        Description
    name_I_begin_ratio_log : str, optional
        Description
    pr_sigma_I_begin : int, optional
        Description
    n_data_points_used : int, optional
        Description
    model : :class:`Cov19Model`
        if none, it is retrieved from the context
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
    mu : :class:`~aesara.tensor.TensorVariable`
    pr_median_delay : float
    pr_sigma_I_begin : float
    n_data_points_used : int

    Returns
    ------------------
    I_begin: :class:`~aesara.tensor.TensorVariable`

    """
    log.info("Uncorrelated prior_E")
    model = modelcontext(model)

    num_regions = () if model.sim_ndim == 1 else model.sim_shape[1:]

    num_new_E_ref = (
        np.nansum(model.new_cases_obs[:n_data_points_used], axis=0) / model.data_len
    )

    diff_E_begin_L1_log = pm.Normal(
        f"{name_E_begin_ratio_log}_L1", mu=0, sigma=pr_sigma_E_begin, shape=len_time
    )

    if model.sim_ndim == 1:
        diff_E_begin_L2_log = diff_E_begin_L1_log
    else:
        sigma_E_begi_log = pm.HalfNormal(
            f"sigma_{name_E_begin_ratio_log}_L1", pr_sigma_E_begin, shape=len_time
        )
        diff_E_begin_L2_log = pm.Normal(
            f"{name_E_begin_ratio_log}_L2_raw",
            mu=0,
            sigma=1,
            shape=((len_time,) + tuple(num_regions)),
        )
        if model.sim_ndim == 2:

            diff_E_begin_L2_log = (
                diff_E_begin_L2_log * sigma_E_begi_log[:, None]
                + diff_E_begin_L1_log[:, None]
            )
        elif model.sim_ndim == 3:
            diff_E_begin_L2_log = (
                diff_E_begin_L2_log * sigma_E_begi_log[:, None, None]
                + diff_E_begin_L1_log[:, None, None]
            )

    new_E_begin = num_new_E_ref * at.exp(diff_E_begin_L2_log)

    pm.Deterministic(name_E_begin, new_E_begin)
    return new_E_begin


def SIR_variants(
    lambda_t_log,
    mu,
    f,
    pr_I_begin=100,
    name_new_I_tv="new_I_tv",
    name_I_begin="I_begin_v",
    name_I_tv="I_tv",
    name_S_t="S_t",
    model=None,
    return_all=False,
    num_variants=5,
    Phi=None,
):
    r"""

    Parameters
    ----------
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
        Time series of the logarithm of the spreading rate. Shape: (time)
    mu : :class:`~aesara.tensor.TensorVariable`
        The recovery rate :math:`\mu`, typically a random variable.
    f : :class:`~aesara.tensor.TensorVariable`
        TODO
    pr_I_begin : float or array_like or :class:`~aesara.tensor.Variable`
        Prior beta of the Half-Cauchy distribution of :math:`I(0)`.
        if type is ``at.Constant``, I_begin will not be inferred by pymc3.
    model : :class:`Cov19Model`
        if none, it is retrieved from the context
    num_variants : number,
        The number of input variants, corresponding to the shape of f.
    Phi : array
        The influx array which is added each timestep should have the shape (variants, time)

    Returns
    ------------------
    new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.
    I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the infected (if return_all set to True)
    S_t : :class:`~aesara.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)
    """

    log.info("SIR with variants")
    model = modelcontext(model)

    # Prior distributions of starting populations (infectious, susceptibles)
    if isinstance(pr_I_begin, at.Variable):
        I_begin = pr_I_begin
    else:
        I_begin = pm.HalfCauchy(name=name_I_begin, beta=pr_I_begin, shape=num_variants)

    # Total number of people in population
    N = model.N_population

    # Initial compartment Suceptible
    S_begin = N - I_begin.sum()

    lambda_t = at.exp(lambda_t_log)

    # Set prior for Influx phi
    if Phi is None:
        loop_phi = False
        Phi = at.zeros(model.sim_length)
    elif isinstance(Phi, at.Variable):
        loop_phi = True

    def next_day(lambda_t, Phi, S_t, I_tv, _, mu, f, N):
        # Variants SIR
        """
        lambda_t.tag.test_value = 1.5
        I_tv.tag.test_value = [50,50,50,50,50]
        S_t.tag.test_value = 5000
        f.tag.test_value = [1,1,1,1,1]
        """

        new_I_tv = f * I_tv * lambda_t * S_t / N

        # Add influx if defined
        if loop_phi:
            new_I_tv += Phi

        # Update new compartments
        I_tv = I_tv + new_I_tv - mu * I_tv
        S_t = S_t - new_I_tv.sum()

        # for stability
        I_tv = at.clip(I_tv, -1, N)
        S_t = at.clip(S_t, 0, N)
        return S_t, I_tv, new_I_tv

    # aesara scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, I, new_I
    new_I_0 = at.zeros_like(I_begin)
    outputs, _ = scan(
        fn=next_day,
        sequences=[lambda_t, Phi],
        outputs_info=[S_begin, I_begin, new_I_0],
        non_sequences=[mu, f, N],
    )
    S_t, I_tv, new_I_tv = outputs
    pm.Deterministic(name_new_I_tv, new_I_tv)
    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_I_tv is not None:
        pm.Deterministic(name_I_tv, I_tv)

    if return_all:
        return new_I_tv, I_tv, S_t
    else:
        return new_I_tv


def kernelized_spread_variants(
    R_t_log,
    f,
    num_variants=5,
    name_new_I_tv="new_I_tv",
    name_new_E_tv="new_E_tv",
    name_S_t="S_t",
    name_new_E_begin="new_E_begin_v",
    name_median_incubation="median_incubation_v",
    pr_new_E_begin=50,
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    model=None,
    return_all=False,
    Phi=None,
    f_is_varying=False,
):
    r"""
    Implements a model similar to the susceptible-exposed-infected-recovered model.
    Instead of a exponential decaying incubation period, the length of the period is
    lognormal distributed.

    Parameters
    ----------
    R_t_log : :class:`~aesara.tensor.TensorVariable`
        time series of the logarithm of the reproduction time, 1 or 2-dimensional. If 2-dimensional, the first
        dimension is time.

    f : :class:`~aesara.tensor.TensorVariable`
        The factor by which the reproduction number of each variant is multiplied by,
        is of shape num_variants

    name_new_E_tv : str, optional
        Name of the ``new_E_tv`` variable

    name_new_I_tv : str, optional
        Name of the ``new_I_tv`` variable

    name_S_t : str, optional
        Name of the ``S_t`` variable

    name_new_E_begin : str, optional
        Name of the ``new_E_begin`` variable

    name_median_incubation : str
        The name under which the median incubation time is saved in the trace

    pr_I_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`I(0)`.
        if type is ``at.Variable``, ``I_begin`` will be set to the provided prior as
        a constant.

    pr_new_E_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`E(0)`.

    pr_mean_median_incubation :
        Prior mean of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        Defaults to 4 days [Nishiura2020]_, which is the median serial interval (the
        important measure here is not exactly the incubation period, but the delay
        until a person becomes infectious which seems to be about 1 day earlier as
        showing symptoms).

    pr_sigma_median_incubation : number or None
        Prior sigma of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        If None, the incubation time will be fixed to the value of
        ``pr_mean_median_incubation`` instead of a random variable
        Default is 1 day.

    sigma_incubation :
        Scale parameter of the :class:`~pymc3.distributions.continuous.Lognormal`
        distribution of the incubation time/ delay until infectiousness. The default
        is set to 0.4, which is about the scale found in [Nishiura2020]_,
        [Lauer2020]_.

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    return_all : bool
        if True, returns ``name_new_I_t``, ``name_new_E_t``,  ``name_I_t``,
        ``name_S_t`` otherwise returns only ``name_new_I_t``

    Phi : array
        The influx array which is added each timestep should have the shape (time, variants)
    f_is_varying : bool
        Whether f varies over time. In this case it is assumed to have shape (time, variants)

    Returns
    -------
    new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.

    new_E_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly exposed persons. (if return_all set to
        True)

    S_t : :class:`~aesara.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)

    """
    log.info("Kernelized spread with variants")
    model = modelcontext(model)

    # Build prior distrubutions:
    # --------------------------

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if isinstance(pr_new_E_begin, at.Variable):
        new_E_begin = pr_new_E_begin
    else:
        new_E_begin = pm.HalfCauchy(
            name=name_new_E_begin,
            beta=pr_new_E_begin,
            shape=(11, num_variants),
        )

    if pr_sigma_median_incubation is None:
        median_incubation = pr_mean_median_incubation
    else:
        median_incubation = pm.Normal(
            name_median_incubation,
            mu=pr_mean_median_incubation,
            sigma=pr_sigma_median_incubation,
        )

    # Set prior for Influx phi
    if Phi is None:
        loop_phi = False
        Phi = at.zeros(model.sim_len)
    elif isinstance(Phi, at.Variable):
        loop_phi = True

    # prepare f
    if not f_is_varying:
        f *= at.ones((model.sim_len, 1))

    # Total number of people in population
    N = model.N_population

    # Starting suceptible pool
    S_begin = N - new_E_begin.sum().sum()

    R_t = at.exp(R_t_log)

    new_I_0 = at.zeros(num_variants)

    # Choose transition rates (E to I) according to incubation period distribution
    x = np.arange(1, 11)[:, None]

    beta = ut.tt_lognormal(x, at.log(median_incubation), sigma_incubation)

    # Runs kernelized spread model with variants:
    def next_day(
        R_t,
        Phi,
        f,
        S_t,
        nE1,
        nE2,
        nE3,
        nE4,
        nE5,
        nE6,
        nE7,
        nE8,
        nE9,
        nE10,
        _,
        beta,
        N,
    ):
        new_I_tv = (
            beta[0] * nE1
            + beta[1] * nE2
            + beta[2] * nE3
            + beta[3] * nE4
            + beta[4] * nE5
            + beta[5] * nE6
            + beta[6] * nE7
            + beta[7] * nE8
            + beta[8] * nE9
            + beta[9] * nE10
        )

        if loop_phi:
            new_I_tv += Phi

        new_E_tv = f * new_I_tv * S_t * R_t / N

        S_t = S_t - new_E_tv.sum()
        S_t = at.clip(S_t, -1, N)

        return S_t, new_E_tv, new_I_tv

    # aesara scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, E's, new_I

    outputs, _ = scan(
        fn=next_day,
        sequences=[R_t, Phi, f],
        outputs_info=[
            S_begin,
            dict(initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]),
            new_I_0,
        ],
        non_sequences=[beta, N],
        # mode="DebugMode",
    )

    # Unzip outputs
    S_t, new_E_tv, new_I_tv = outputs

    # Add new_cases to trace
    pm.Deterministic(name_new_I_tv, new_I_tv)

    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_new_E_tv is not None:
        pm.Deterministic(name_new_E_tv, new_E_tv)

    if return_all:
        return new_I_tv, new_E_tv, S_t
    else:
        return new_I_tv


def kernelized_spread_gender(
    lambda_t_log,
    gender_interaction_matrix,
    name_new_I_t="new_I_t",
    name_new_E_t="new_E_t",
    name_S_t="S_t",
    name_new_E_begin="new_E_begin",
    name_median_incubation="median_incubation",
    pr_new_E_begin=50,
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    num_gender=2,
    model=None,
    return_all=False,
):
    r"""
    Implements a model similar to the susceptible-exposed-infected-recovered model.
    Instead of a exponential decaying incubation period, the length of the period is
    lognormal distributed.

    Parameters
    ----------
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
        Time series of the logarithm of the spreading rate, 2 or 3-dimensional. If 3-dimensional, the first
        dimension is time.
        shape: time, gender, [country]

    gender_interaction_matrix : :class:`~aesara.tensor.TensorVariable`
        Gender interaction matrix should be of shape (num_gender,num_gender)

    name_new_I_t : str, optional
        Name of the ``new_I_t`` variable

    name_S_t : str, optional
        Name of the ``S_t`` variable

    name_new_E_begin : str, optional
        Name of the ``new_E_begin`` variable

    name_median_incubation : str
        The name under which the median incubation time is saved in the trace

    pr_I_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`I(0)`.
        if type is ``at.Variable``, ``I_begin`` will be set to the provided prior as
        a constant.

    pr_new_E_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`E(0)`.

    pr_mean_median_incubation :
        Prior mean of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        Defaults to 4 days [Nishiura2020]_, which is the median serial interval (the
        important measure here is not exactly the incubation period, but the delay
        until a person becomes infectious which seems to be about 1 day earlier as
        showing symptoms).

    pr_sigma_median_incubation : number or None
        Prior sigma of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        If None, the incubation time will be fixed to the value of
        ``pr_mean_median_incubation`` instead of a random variable
        Default is 1 day.

    sigma_incubation :
        Scale parameter of the :class:`~pymc3.distributions.continuous.Lognormal`
        distribution of the incubation time/ delay until infectiousness. The default
        is set to 0.4, which is about the scale found in [Nishiura2020]_,
        [Lauer2020]_.

    pr_sigma_mu : float or array_like
        Prior for the sigma of the lognormal distribution of recovery rate
        :math:`\mu`.

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    return_all : bool
        if True, returns ``name_new_I_t``, ``name_new_E_t``,  ``name_I_t``,
        ``name_S_t`` otherwise returns only ``name_new_I_t``
    num_gender : int, optional
        Number of proposed gender groups (dimension size)

    Returns
    -------
    name_new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.

    name_new_E_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly exposed persons. (if return_all set to
        True)

    name_S_t : :class:`~aesara.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)

    """
    log.info("kernelized spread")
    model = modelcontext(model)

    # Build prior distrubutions:
    # --------------------------

    # Total number of people in population
    N = model.N_population  # shape: [gender, country]

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if isinstance(pr_new_E_begin, at.Variable):
        new_E_begin = pr_new_E_begin
    else:
        if not model.is_hierarchical:
            new_E_begin = pm.HalfCauchy(
                name=name_new_E_begin, beta=pr_new_E_begin, shape=(11, num_gender)
            )
        else:
            new_E_begin = pm.HalfCauchy(
                name=name_new_E_begin,
                beta=pr_new_E_begin,
                shape=(11, num_gender, model.shape_of_regions),
            )

    # shape: countries
    S_begin = N - pm.math.sum(new_E_begin, axis=(0, 1))

    lambda_t = at.exp(lambda_t_log)
    # shape: genders, countries
    new_I_0 = at.zeros(model.sim_shape[1:])

    if pr_sigma_median_incubation is None:
        median_incubation = pr_mean_median_incubation
    else:
        median_incubation = pm.Normal(
            name_median_incubation,
            mu=pr_mean_median_incubation,
            sigma=pr_sigma_median_incubation,
        )

    # Choose transition rates (E to I) according to incubation period distribution
    if not model.is_hierarchical:
        x = np.arange(1, 11)
    else:
        x = np.arange(1, 11)[:, None]

    beta = ut.tt_lognormal(x, at.log(median_incubation), sigma_incubation)

    # Runs kernelized spread model:
    def next_day(
        lambda_t,
        S_t,
        nE1,
        nE2,
        nE3,
        nE4,
        nE5,
        nE6,
        nE7,
        nE8,
        nE9,
        nE10,
        _,
        beta,
        N,
        gender_interaction_matrix,
    ):
        new_I_t = (
            beta[0] * nE1
            + beta[1] * nE2
            + beta[2] * nE3
            + beta[3] * nE4
            + beta[4] * nE5
            + beta[5] * nE6
            + beta[6] * nE7
            + beta[7] * nE8
            + beta[8] * nE9
            + beta[9] * nE10
        )
        print(new_I_t.shape)  #
        # shape: gender, country
        new_E_t = lambda_t / N[None, :] * new_I_t * S_t[None, :]

        # Interaction between gender groups (gender,gender)@(gender,countries)
        new_E_t = at.dot(gender_interaction_matrix, new_E_t)

        # Update suceptible compartement
        S_t = S_t - new_E_t.sum(axis=0)
        S_t = at.clip(S_t, -1, N)
        return S_t, new_E_t, new_I_t

    # aesara scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, E's, new_I
    outputs, _ = scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[
            S_begin,  # shape: countries
            dict(
                initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
            ),  # shape time, gender, countries
            new_I_0,  # shape gender, countries
        ],
        non_sequences=[beta, N, gender_interaction_matrix],
    )
    S_t, new_E_t, new_I_t = outputs
    pm.Deterministic(name_new_I_t, new_I_t)

    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_new_E_t is not None:
        pm.Deterministic(name_new_E_t, new_E_t)

    if return_all:
        return new_I_t, new_E_t, S_t
    else:
        return new_I_t


def kernelized_spread_with_interaction(
    R_t_log,
    interaction_matrix,
    num_groups,
    influx=None,
    name_new_I_t="new_I_t",
    name_new_E_t="new_E_t",
    name_S_t="S_t",
    name_new_E_begin="new_E_begin",
    name_median_incubation="median_incubation",
    pr_new_E_begin=50,
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    model=None,
    return_all=False,
):
    r"""
    Implements a model similar to the susceptible-exposed-infected-recovered model.
    Instead of a exponential decaying incubation period, the length of the period is
    lognormal distributed. In this model, we have an interaction between different
    groups, could be different age-groups, countries, states,...

    Parameters
    ----------
    R_t_log : :class:`~aesara.tensor.TensorVariable`
        Time series of the logarithm of the spreading rate, 2 or 3-dimensional. The first
        dimension is time.
        shape: time, num_groups, [independent dimension]

    interaction_matrix : :class:`~aesara.tensor.TensorVariable`
        Interaction matrix should be of shape (num_groups, num_groups)

    num_groups : int
         number of groups

    name_new_I_t : str, optional
        Name of the ``new_I_t`` variable

    name_S_t : str, optional
        Name of the ``S_t`` variable

    name_new_E_begin : str, optional
        Name of the ``new_E_begin`` variable

    name_median_incubation : str
        The name under which the median incubation time is saved in the trace

    pr_I_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`I(0)`.
        if type is ``at.Variable``, ``I_begin`` will be set to the provided prior as
        a constant.

    pr_new_E_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`E(0)`.

    pr_mean_median_incubation :
        Prior mean of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        Defaults to 4 days [Nishiura2020]_, which is the median serial interval (the
        important measure here is not exactly the incubation period, but the delay
        until a person becomes infectious which seems to be about 1 day earlier as
        showing symptoms).

    pr_sigma_median_incubation : number or None
        Prior sigma of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        If None, the incubation time will be fixed to the value of
        ``pr_mean_median_incubation`` instead of a random variable
        Default is 1 day.

    sigma_incubation :
        Scale parameter of the :class:`~pymc3.distributions.continuous.Lognormal`
        distribution of the incubation time/ delay until infectiousness. The default
        is set to 0.4, which is about the scale found in [Nishiura2020]_,
        [Lauer2020]_.

    pr_sigma_mu : float or array_like
        Prior for the sigma of the lognormal distribution of recovery rate
        :math:`\mu`.

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    return_all : bool
        if True, returns ``name_new_I_t``, ``name_new_E_t``,  ``name_I_t``,
        ``name_S_t`` otherwise returns only ``name_new_I_t``

    Returns
    -------
    name_new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.

    name_new_E_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly exposed persons. (if return_all set to
        True)

    name_S_t : :class:`~aesara.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)

    """
    log.info("kernelized spread")
    model = modelcontext(model)

    # Total number of people in population
    N = model.N_population  # shape: [num_groups]

    assert len(N) == num_groups
    assert model.sim_shape[-1] == num_groups
    assert len(model.sim_shape) == 2

    # Build prior distrubutions:
    # --------------------------

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if isinstance(pr_new_E_begin, at.Variable):
        new_E_begin = pr_new_E_begin
    else:
        new_E_begin = pm.HalfCauchy(
            name=name_new_E_begin,
            beta=pr_new_E_begin,
            shape=(11, num_groups),
        )

    # shape: num_groups
    S_begin = N - pm.math.sum(new_E_begin, axis=(0,))

    R_t = at.exp(R_t_log)
    # shape: num_groups, [independent dimension]
    new_I_0 = at.zeros(model.sim_shape[1:])
    if influx is None:
        influx = at.zeros(model.sim_shape)

    if pr_sigma_median_incubation is None:
        median_incubation = pr_mean_median_incubation
    else:
        median_incubation = pm.Normal(
            name_median_incubation,
            mu=pr_mean_median_incubation,
            sigma=pr_sigma_median_incubation,
        )

    # Choose transition rates (E to I) according to incubation period distribution

    x = np.arange(1, 11)[:, None]

    beta = ut.tt_lognormal(x, at.log(median_incubation), sigma_incubation)

    # Runs kernelized spread model:
    def next_day(
        R_t,
        influx_t,
        S_t,
        nE1,
        nE2,
        nE3,
        nE4,
        nE5,
        nE6,
        nE7,
        nE8,
        nE9,
        nE10,
        _,
        beta,
        N,
        interaction_matrix,
    ):
        new_I_t = (
            beta[0] * nE1
            + beta[1] * nE2
            + beta[2] * nE3
            + beta[3] * nE4
            + beta[4] * nE5
            + beta[5] * nE6
            + beta[6] * nE7
            + beta[7] * nE8
            + beta[8] * nE9
            + beta[9] * nE10
        )

        # The reproduction number is assumed to have a symmetric effect, hence the sqrt
        new_E_t = at.sqrt(R_t) / N * new_I_t * S_t

        # Interaction between gender groups (groups,groups)@(groups, [evtl. other dimension])

        new_E_t = at.sqrt(R_t) * at.dot(interaction_matrix, new_E_t) + influx_t

        # Update suceptible compartement
        S_t = S_t - new_E_t
        S_t = at.clip(S_t, -1, N)
        return S_t, new_E_t, new_I_t

    # aesara scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, E's, new_I
    outputs, _ = scan(
        fn=next_day,
        sequences=[R_t, influx],
        outputs_info=[
            S_begin,  # shape: countries
            dict(
                initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
            ),  # shape time, groups, independent dimension
            new_I_0,  # shape groups, independent dimension
        ],
        non_sequences=[beta, N, interaction_matrix],
    )
    S_t, new_E_t, new_I_t = outputs

    if name_new_I_t is not None:
        pm.Deterministic(name_new_I_t, new_I_t)

    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_new_E_t is not None:
        pm.Deterministic(name_new_E_t, new_E_t)

    if return_all:
        return new_I_t, new_E_t, S_t
    else:
        return new_I_t


def kernelized_spread_tmp(
    R_t_log,
    name_new_I_t="new_I_t",
    name_new_E_t="new_E_t",
    name_S_t="S_t",
    name_new_E_begin="new_E_begin",
    name_median_incubation="median_incubation",
    pr_new_E_begin=50,
    pr_mean_median_incubation=4,
    pr_sigma_median_incubation=1,
    sigma_incubation=0.4,
    model=None,
    return_all=False,
):
    r"""
    Implements a model similar to the susceptible-exposed-infected-recovered model.
    Instead of a exponential decaying incubation period, the length of the period is
    lognormal distributed.

    Parameters
    ----------
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
        Time series of the logarithm of the spreading rate, 2 or 3-dimensional. If 3-dimensional, the first
        dimension is time.
        shape: time, gender, [country]

    gender_interaction_matrix : :class:`~aesara.tensor.TensorVariable`
        Gender interaction matrix should be of shape (num_gender,num_gender)

    name_new_I_t : str, optional
        Name of the ``new_I_t`` variable

    name_S_t : str, optional
        Name of the ``S_t`` variable

    name_new_E_begin : str, optional
        Name of the ``new_E_begin`` variable

    name_median_incubation : str
        The name under which the median incubation time is saved in the trace

    pr_I_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`I(0)`.
        if type is ``at.Variable``, ``I_begin`` will be set to the provided prior as
        a constant.

    pr_new_E_begin : float or array_like
        Prior beta of the :class:`~pymc3.distributions.continuous.HalfCauchy`
        distribution of :math:`E(0)`.

    pr_mean_median_incubation :
        Prior mean of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        Defaults to 4 days [Nishiura2020]_, which is the median serial interval (the
        important measure here is not exactly the incubation period, but the delay
        until a person becomes infectious which seems to be about 1 day earlier as
        showing symptoms).

    pr_sigma_median_incubation : number or None
        Prior sigma of the :class:`~pymc3.distributions.continuous.Normal`
        distribution of the median incubation delay  :math:`d_{\text{incubation}}`.
        If None, the incubation time will be fixed to the value of
        ``pr_mean_median_incubation`` instead of a random variable
        Default is 1 day.

    sigma_incubation :
        Scale parameter of the :class:`~pymc3.distributions.continuous.Lognormal`
        distribution of the incubation time/ delay until infectiousness. The default
        is set to 0.4, which is about the scale found in [Nishiura2020]_,
        [Lauer2020]_.

    pr_sigma_mu : float or array_like
        Prior for the sigma of the lognormal distribution of recovery rate
        :math:`\mu`.

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    return_all : bool
        if True, returns ``name_new_I_t``, ``name_new_E_t``,  ``name_I_t``,
        ``name_S_t`` otherwise returns only ``name_new_I_t``
    num_gender : int, optional
        Number of proposed gender groups (dimension size)

    Returns
    -------
    name_new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.

    name_new_E_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly exposed persons. (if return_all set to
        True)

    name_S_t : :class:`~aesara.tensor.TensorVariable`
        time series of the susceptible (if return_all set to True)

    """
    log.info("kernelized spread")
    model = modelcontext(model)

    # Total number of people in population
    N = model.N_population  # shape: ()

    assert len(model.sim_shape) == 1

    # Build prior distrubutions:
    # --------------------------

    # Prior distributions of starting populations (exposed, infectious, susceptibles)
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if isinstance(pr_new_E_begin, at.Variable):
        new_E_begin = pr_new_E_begin
    else:
        new_E_begin = pm.HalfCauchy(
            name=name_new_E_begin,
            beta=pr_new_E_begin,
            shape=(11,),
        )

    # shape: num_groups
    S_begin = N - pm.math.sum(new_E_begin, axis=(0,))

    R_t = at.exp(R_t_log)

    new_I_0 = at.zeros(())

    if pr_sigma_median_incubation is None:
        median_incubation = pr_mean_median_incubation
    else:
        median_incubation = pm.Normal(
            name_median_incubation,
            mu=pr_mean_median_incubation,
            sigma=pr_sigma_median_incubation,
        )

    # Choose transition rates (E to I) according to incubation period distribution

    x = np.arange(1, 11)[:, None]

    beta = ut.tt_lognormal(x, at.log(median_incubation), sigma_incubation)

    # Runs kernelized spread model:
    def next_day(
        R_t,
        S_t,
        nE1,
        nE2,
        nE3,
        nE4,
        nE5,
        nE6,
        nE7,
        nE8,
        nE9,
        nE10,
        _,
        beta,
        N,
    ):
        new_I_t = (
            beta[0] * nE1
            + beta[1] * nE2
            + beta[2] * nE3
            + beta[3] * nE4
            + beta[4] * nE5
            + beta[5] * nE6
            + beta[6] * nE7
            + beta[7] * nE8
            + beta[8] * nE9
            + beta[9] * nE10
        )
        print(new_I_t.shape)  #
        # shape: gender, country
        new_E_t = R_t / N * new_I_t * S_t

        # Update susceptible compartement
        S_t = S_t - new_E_t.sum(axis=0)
        S_t = at.clip(S_t, -1, N)
        return S_t, new_E_t, new_I_t

    # aesara scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, E's, new_I
    outputs, _ = scan(
        fn=next_day,
        sequences=[R_t],
        outputs_info=[
            S_begin,  # shape: countries
            dict(
                initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
            ),  # shape time, gender, countries
            new_I_0,  # shape gender, countries
        ],
        non_sequences=[beta, N],
    )
    S_t, new_E_t, new_I_t = outputs
    pm.Deterministic(name_new_I_t, new_I_t)

    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_new_E_t is not None:
        pm.Deterministic(name_new_E_t, new_E_t)

    if return_all:
        return new_I_t, new_E_t, S_t
    else:
        return new_I_t
