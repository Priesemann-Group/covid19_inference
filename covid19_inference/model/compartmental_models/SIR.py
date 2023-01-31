import logging
import numpy as np
import pymc as pm

from aesara import scan
import aesara.tensor as at

from ..model import *

log = logging.getLogger(__name__)


def SIR(
    lambda_t_log,
    # mu
    mu=None,
    mu_kwargs={
        "name": "mu",
        "mu": np.log(1 / 8),
        "sigma": 0.2,
    },
    # I_begin
    I_begin=None,
    I_begin_kwargs={
        "name": "I_begin",
        "beta": 100,
    },
    # Other
    name_new_I_t="new_I_t",
    name_I_t="I_t",
    name_S_t="S_t",
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

    mu : None or :class:`pymc.Continuous`
        Distribution of the recovery rate :math:`\mu`. Defaults to
        :class:`pymc.LogNormal` with the arguments defined
        in ``mu_kwargs``. Can be 0 or 1-dimensional. If 1-dimensional, the dimension
        are the different regions.

    mu_kwargs : dict
        Arguments for the recovery rate distribution. Defaults to ``{"name": "mu", "mu":log(1/8), "sigma":0.2}``. See :class:`pymc.LogNormal` for more
        options. If no shape is given, the shape is inferred from the model.

    I_begin : None or :class:`~aesara.tensor.TensorVariable`
        Distribution for initial value of infected pool i.e. :math:`I(0)`. Defaults to
        :class:`pymc.HalfCauchy` with the arguments defined
        in ``I_begin_kwargs``. Can be 0 or 1-dimensional. If 1-dimensional,
        the dimension are the different regions.

    I_begin_kwargs : dict
        Arguments for the initial value of infected pool distribution. Defaults to
        ``{"name": "I_begin", "beta": 100}``. See :class:`pymc.HalfCauchy`
        for more options. If no shape is given, the shape is inferred from the model.

    Other Parameters
    ----------------
    name_new_I_t : str, optional
        Name of the ``new_I_t`` variable in the trace, set to None to avoid adding as
        trace variable. Defaults to ``new_I_t``.

    name_I_t : str, optional
        Name of the ``I_t`` variable in the trace, set to None to avoid adding as trace
        variable. Defaults to ``I_t``.

    name_S_t : str, optional
        Name of the ``S_t`` variable in the trace, set to None to avoid adding as trace
        variable. Defaults to ``S_t``.

    model : :class:`Cov19Model`, optional
        Is retrieved from the context by default.

    return_all : bool, optional
        If True, returns ``new_I_t``, ``I_t``, ``S_t`` tensors otherwise returns only ``new_I_t``
        tensor.

    Returns
    ------------------
    new_I_t : :class:`~aesara.tensor.TensorVariable`
        time series of the number daily newly infected persons.
    I_t : :class:`~aesara.tensor.TensorVariable`, optional
        time series of the infected (if return_all set to True)
    S_t : :class:`~aesara.tensor.TensorVariable`, optional
        time series of the susceptible (if return_all set to True)

    """
    log.info("Compartmental Model (SIR)")
    model = modelcontext(model)

    # Defaults for mu
    if mu is None:
        shape = mu_kwargs.pop("shape", model.shape_of_regions)
        mu = pm.LogNormal(**mu_kwargs, shape=shape)

    # Defaults for I_begin
    if I_begin is None:
        shape = I_begin_kwargs.pop("shape", model.shape_of_regions)
        I_begin = pm.HalfCauchy(**I_begin_kwargs, shape=shape)

    # Initial values
    N = model.N_population  # Total number of people in population
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

    # Add to trace
    if name_new_I_t is not None:
        pm.Deterministic(name_new_I_t, new_I_t)
    if name_S_t is not None:
        pm.Deterministic(name_S_t, S_t)
    if name_I_t is not None:
        pm.Deterministic(name_I_t, I_t)

    if return_all:
        return new_I_t, I_t, S_t
    else:
        return new_I_t
