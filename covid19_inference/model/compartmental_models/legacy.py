import logging
import numpy as np
import pymc as pm

from aesara import scan
import aesara.tensor as at

from ..model import *
from .. import utility as ut

log = logging.getLogger(__name__)

import warnings
import functools


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


@deprecated
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
        Time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional
        the first dimension is time.

    mu : None or :class:`~pymc.distribution.Continuous`, optional
        Distribution of the recovery rate :math:`\mu`. Defaults to
        :class:`~pymc.distributions.continuous.Lognormal` with the arguments defined
        in ``mu_kwargs``. Can be 0 or 1-dimensional. If 1-dimensional, the dimension
        are the different regions.

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

    log.info("Compartmental Model (SIR with variants)")
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


@deprecated
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


@deprecated
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


@deprecated
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


@deprecated
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
