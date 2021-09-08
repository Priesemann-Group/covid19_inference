# ------------------------------------------------------------------------------ #
# Helper functions that are used by other parts of the modeling
# ------------------------------------------------------------------------------ #

import platform
import logging
import theano
import theano.tensor as tt
import pymc3 as pm

from .model import *

log = logging.getLogger(__name__)

# workaround for macos, sufficient to do this once
if platform.system() == "Darwin":
    theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"


def hierarchical_normal(
    pr_mean,
    pr_sigma,
    name_L1="delay_hc_L1",
    name_L2="delay_hc_L2",
    name_sigma="delay_hc_sigma",
    model=None,
    error_fact=2.0,
    error_cauchy=True,
    shape=None,
):
    r"""
    Implements an hierarchical normal model:

    .. math::

        x_\text{L1} &= Normal(\text{pr\_mean}, \text{pr\_sigma})\\
        y_{i, \text{L2}} &= Normal(x_\text{L1}, \sigma_\text{L2})\\
        \sigma_\text{L2} &= HalfCauchy(\text{error\_fact} \cdot \text{pr\_sigma})

    It is however implemented in a non-centered way, that the second line is changed to:

     .. math::

        y_{i, \text{L2}} &= x_\text{L1} +  Normal(0,1) \cdot \sigma_\text{L2}

    See for example https://arxiv.org/pdf/1312.0906.pdf


    Parameters
    ----------
    name_L1 : str
        Name under which :math:`x_\text{L1}` is saved in the trace.
    name_L2 : str
        Name under which :math:`x_\text{L2}` is saved in the trace. The non-centered distribution in addition
        saved with a suffix _raw added.
    name_sigma : str
        Name under which :math:`\sigma_\text{L2}` is saved in the trace.
    pr_mean : float
        Prior mean of :math:`x_\text{L1}`
    pr_sigma : float
        Prior sigma for :math:`x_\text{L1}` and (muliplied by ``error_fact``) for :math:`\sigma_\text{L2}`
    len_L2 : int
        length of :math:`y_\text{L2}`
    error_fact : float
        Factor by which ``pr_sigma`` is multiplied as prior for `\sigma_\text{L2}`
    error_cauchy : bool
        if False, a :math:`HalfNormal` distribution is used for :math:`\sigma_\text{L2}` instead of :math:`HalfCauchy`

    Returns
    -------
    y : :class:`~theano.tensor.TensorVariable`
        the random variable :math:`y_\text{L2}`
    x : :class:`~theano.tensor.TensorVariable`
        the random variable :math:`x_\text{L1}`
    """

    model = modelcontext(model)

    if shape is None:
        shape = model.shape_of_regions

    if not model.is_hierarchical:
        raise RuntimeError("Model is not hierarchical.")

    if error_cauchy:
        sigma_Y = (
            (pm.HalfCauchy(name_sigma, beta=1, transform=pm.transforms.log_exp_m1))
            * error_fact
            * pr_sigma
        )
    else:
        sigma_Y = (
            (pm.HalfNormal(name_sigma, sigma=1, transform=pm.transforms.log_exp_m1))
            * error_fact
            * pr_sigma
        )

    X = pm.Normal(name_L1, mu=pr_mean, sigma=pr_sigma)
    phi = pm.Normal(
        name_L2 + "_raw_", mu=0, sigma=1, shape=shape
    )  # (1-w**2)*sigma_X+1*w**2, shape=len_Y)
    Y = X + phi * sigma_Y
    pm.Deterministic(name_L2, Y)

    return Y, X


# utility.py
# names
# still do decide
def hierarchical_beta(name, name_sigma, pr_mean, pr_sigma, len_L2, model=None):

    model = modelcontext(model)

    if not model.is_hierarchical:  # not hierarchical
        Y = pm.Beta(name, alpha=pr_mean / pr_sigma, beta=1 / pr_sigma * (1 - pr_mean))
        X = None
    else:
        sigma_Y = pm.HalfCauchy(name_sigma + "_hc_L2", beta=pr_sigma)
        X = pm.Beta(
            name + "_hc_L1", alpha=pr_mean / pr_sigma, beta=1 / pr_sigma * (1 - pr_mean)
        )
        Y = pm.Beta(
            name + "_hc_L2", alpha=X / sigma_Y, beta=1 / sigma_Y * (1 - X), shape=len_L2
        )

    return Y, X


# utility.py
def tt_lognormal(x, mu, sigma):
    """
    Calculates a lognormal pdf for integer spaced x input.
    """
    x = tt.nnet.relu(x - 1e-12) + 1e-12  # clip values at 1e-12
    distr = 1 / x * tt.exp(-((tt.log(x) - mu) ** 2) / (2 * sigma ** 2))

    # normalize, add a small offset in case the sum is zero
    return distr / tt.sum(distr, axis=0) + 1e-8


def tt_gamma(x, mu=None, sigma=None, alpha=None, beta=None):
    """
    Calculates a gamma distribution pdf for integer spaced x input. Parametrized similarly to
    :class:`~pymc3.distributions.continuous.Gamma`
    """
    assert (alpha is None and beta is None) != (mu is None and sigma is None)
    if alpha is None and beta is None:
        alpha = mu ** 2 / (sigma ** 2 + 1e-8)
        beta = mu / (sigma ** 2 + 1e-8)
    x = tt.nnet.relu(x - 1e-12) + 1e-12  # clip values at 1e-12
    distr = beta ** alpha * x ** (alpha - 1) * tt.exp(-beta * x)

    # normalize, add a small offset in case the sum is zero
    return distr / (tt.sum(distr, axis=0) + 1e-8)
