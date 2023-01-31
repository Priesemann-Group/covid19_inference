import logging
import numpy as np
import pymc as pm

from aesara import scan
import aesara.tensor as at

from ..model import *
from .. import utility as ut

log = logging.getLogger(__name__)


def SEIR(
    lambda_t_log,
    # mu
    mu=None,
    mu_kwargs={
        "name": "mu",
        "mu": np.log(1 / 8),
        "sigma": 0.2,
    },
    I_begin=None,
    I_begin_kwargs={
        "name": "I_begin",
        "beta": 100,
    },
    # new_E_begin
    new_E_begin=None,
    new_E_begin_kwargs={
        "name": "new_E_begin",
        "beta": 50,
    },
    # Median incubation period
    median_incubation=None,
    median_incubation_kwargs={
        "name": "median_incubation",
        "mu": 4,
        "sigma": 1,
    },
    sigma_incubation=0.4,
    # Other
    name_new_I_t="new_I_t",
    name_new_E_t="new_E_t",
    name_I_t="I_t",
    name_S_t="S_t",
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
        Time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional
        the first dimension is time.

    mu : None or :class:`pymc.Continuous`, optional
        Distribution of the recovery rate :math:`\mu`. Defaults to
        :class:`pymc.LogNormal` with the arguments defined
        in ``mu_kwargs``. Can be 0 or 1-dimensional. If 1-dimensional, the dimension
        are the different regions.

    mu_kwargs : dict, optional
        Arguments for the recovery rate distribution. Defaults to ``{"name": "mu", "mu":log(1/8), "sigma":0.2}``. See :class:`pymc.LogNormal` for more
        options. If no shape is given, the shape is inferred from the model. Ignored
        if ``mu`` is not None.

    I_begin : None or :class:`pymc.Continuous`, optional
        Distribution for initial value of infected pool i.e. :math:`I(0)`. Defaults to
        :class:`pymc.HalfCauchy` with the arguments defined
        in ``I_begin_kwargs``. Can be 0 or 1-dimensional. If 1-dimensional,
        the dimension are the different regions.

    I_begin_kwargs : dict, optional
        Arguments for the initial value of infected pool distribution. Defaults to
        ``{"name": "I_begin", "beta": 100}``. See :class:`pymc.HalfCauchy`
        for more options. If no shape is given, the shape is inferred from the model.
        Ignored if ``I_begin`` is not None.

    new_E_begin : None or :class:`pymc.Continuous`, optional
        Distribution for initial value of exposed pool i.e. :math:`E(0)`. Defaults to
        :class:`pymc.HalfCauchy` with the arguments defined
        in ``new_E_begin_kwargs``. Can be 1 or 2-dimensional, the first dimension always
        being a running window of the last 10 days. E.g. ``(11, shape_of_regions)``.

    new_E_begin_kwargs : dict, optional
        Arguments for the initial value of exposed pool distribution. Defaults to
        ``{"name": "new_E_begin", "beta": 50}``. See :class:`pymc.HalfCauchy`
        for more options. If no shape is given, the shape is inferred from the model. E.g.
        (11, shape_of_regions). Ignored if ``new_E_begin`` is not None.

    median_incubation : None or :class:`pymc.Continuous`, optional
        Distribution for the median incubation period :math:`d_{\text{incubation}}`. Defaults to
        :class:`pymc.Normal` with the arguments defined
        in ``median_incubation_kwargs``. Can be 0 or 1-dimensional. If 1-dimensional,
        the dimension are the different regions.

    median_incubation_kwargs : dict, optional
        Arguments for the median incubation period distribution. Defaults to
        ``{"name": "median_incubation", "mu": 4, "sigma": 1}``. See :class:`pymc.Normal`
        for more options. If no shape is given, the shape is inferred from the model.
        Ignored if ``median_incubation`` is not None. mu defaults to 4 days [Nishiura2020]_, which is the median serial interval (the important measure here is not exactly the incubation period, but the delay until a person becomes infectious which seems to be about 1 day earlier as showing symptoms).

    sigma_incubation : number or :class:`pymc.Continuous`, optional
        Scale parameter of the :class:`pymc.LogNormal`
        distribution of the incubation time/ delay until infectiousness. The default
        is set to 0.4, which is about the scale found in [Nishiura2020]_,
        [Lauer2020]_.

    Other Parameters
    ----------------
    name_new_I_t : str, optional
        Name of the ``new_I_t`` variable in the trace, set to None to avoid adding as
        trace variable. Defaults to ``new_I_t``.

    name_new_E_t : str, optional
        Name of the ``new_E_t`` variable in the trace, set to None to avoid adding as trace
        variable. Defaults to ``new_E_t``.

    name_I_t : str, optional
        Name of the ``I_t`` variable in the trace, set to None to avoid adding as trace
        variable. Defaults to ``I_t``.

    name_S_t : str, optional
        Name of the ``S_t`` variable in the trace, set to None to avoid adding as trace
        variable. Defaults to ``S_t``.

    model : :class:`Cov19Model`, optional
        if none, it is retrieved from the context

    return_all : bool, optional
        If True, returns ``new_I_t``, ``new_E_t``, ``I_t``, ``S_t`` tensors otherwise returns
        only ``new_I_t`` tensor.

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
    log.info("Compartmental Model (SEIR)")
    model = modelcontext(model)

    # Defaults for mu
    if mu is None:
        shape = mu_kwargs.pop("shape", model.shape_of_regions)
        mu = pm.LogNormal(**mu_kwargs, shape=shape)

    # Defaults for I_begin
    if I_begin is None:
        shape = I_begin_kwargs.pop("shape", model.shape_of_regions)
        I_begin = pm.HalfCauchy(**I_begin_kwargs, shape=shape)

    # Defaults for new_E_begin
    # We choose to consider the transitions of newly exposed people of the last 10 days.
    if new_E_begin is None:
        if not model.is_hierarchical:
            shape = new_E_begin_kwargs.pop("shape", 11)
        else:
            shape = new_E_begin_kwargs.pop("shape", (11, model.shape_of_regions))
        new_E_begin = pm.HalfCauchy(**new_E_begin_kwargs, shape=shape)

    # Defaults for median incubation
    if median_incubation is None:
        median_incubation = pm.Normal(**median_incubation_kwargs)

    # Initial values
    N = model.N_population  # Total number of people in population
    S_begin = N - I_begin - pm.math.sum(new_E_begin, axis=0)

    lambda_t = at.exp(lambda_t_log)
    new_I_0 = at.zeros_like(I_begin)

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

    # Add to trace
    if name_new_I_t is not None:
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
