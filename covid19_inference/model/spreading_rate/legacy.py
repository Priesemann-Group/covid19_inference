import logging
import pymc as pm
import aesara.tensor as at
from ..model import modelcontext
from ..utility import deprecated

log = logging.getLogger(__name__)


@deprecated
def R_t_log_with_longtailed_dist(
    R_0_log, dist=None, model=None, name_R_t=None, rho_scale_prior=0.1, **dist_priors
):
    r"""
    Builds a time dependent reproduction number :math:`R_t` by using a long tailed distribution.

    .. math::

        log(R_t) &= log(R_0) + cumsum( s \cdot \Delta \rho) \\
        s &\sim \text{HalfCauchy}(\mu=0.1)

    The parameter :math:`\Delta \rho` is dependent on the given distribution. Possible values are
    ``studentT``, ``weibull``, ``cauchy``. In theory you can also pass your another function of type
    :class:`pm.distributions.Continuous`.


    Parameters
    ----------
    R_0_log : number
        Initial value of the reproduction number in log transform
    dist : string or :class:`pm.distributions.Continuous`
        Distribution to use. Possible values are ``studentT``, ``weibull``, ``cauchy``.
        You can also pass your own distribution. If you use your own, be sure to configure the kwargs of the distribution
        using the ``dist_priors`` parameter. The distribution should have the time dimension as first axis.
    dist_priors : dict
        Dictionary of kwargs i.e. mostly priors for the distribution.
    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    Returns
    -------
    R_t_log
    """

    log.info("R_t with long tailed dist")
    # Get our default mode context
    model = modelcontext(model)

    # Parse dist priors
    if "name" not in dist_priors:
        dist_priors["name"] = "Delta_rho"
    if "shape" not in dist_priors:
        dist_priors["shape"] = model.sim_len

    # Helper functions to define the distributions
    def _get_StundentT_dist(**dist_priors):
        if "nu" not in dist_priors:
            dist_priors["nu"] = 1.5
        if "mu" not in dist_priors:
            dist_priors["mu"] = 0
        if "sigma" not in dist_priors and "lam" not in dist_priors:
            dist_priors["sigma"] = 1
        return pm.StudentT(**dist_priors)

    def _get_Weibull_dist(**dist_priors):
        if "alpha" not in dist_priors:
            dist_priors["alpha"] = 0.2
        if "beta" not in dist_priors:
            dist_priors["beta"] = 1
        if "scale" not in dist_priors:
            mean_influx = 1
            scale = mean_influx / scipy.special.gamma(1 + 1 / dist_priors["alpha"])
        else:
            scale = dist_priors["scale"]
            del dist_priors["scale"]
        dist = (
            (pm.Weibull(**dist_priors))
            * scale
            * (pm.Normal("wb_scale", 0, 1, shape=dist_priors["shape"]))
        )
        return dist

    def _get_Cauchy_dist(**dist_priors):
        if "alpha" not in dist_priors:
            dist_priors["alpha"] = 0
        if "beta" not in dist_priors:
            dist_priors["beta"] = 1
        return pm.Cauchy(**dist_priors)

    # Parse dist
    if dist is None:
        dist = "studentT"
    if isinstance(dist, str):
        if dist == "studentT":
            dist = _get_StundentT_dist(**dist_priors)
        elif dist == "weibull":
            dist = _get_Weibull_dist(**dist_priors)
        elif dist == "cauchy":
            dist = _get_Cauchy_dist(**dist_priors)
    elif isinstance(dist, pm.distributions.Continuous):
        dist = dist("", **dist_priors)
    elif not isinstance(dist, pm.distributions.Continuous):
        raise ValueError(f"Distribution {dist} is not supported")

    # Build the model equations
    scale = pm.HalfCauchy("Delta_rho_scale", beta=rho_scale_prior)
    R_t_log = R_0_log + at.cumsum(scale * dist, axis=0)

    # Create responding R_t pymc variable with given name (from parameters)
    if name_R_t is not None:
        pm.Deterministic(name_R_t, at.exp(R_t_log))

    return R_t_log


@deprecated
def _smooth_step_function(start_val, end_val, t_begin, t_end, t_total):
    """
    Instead of going from start_val to end_val in one step, make the change a
    gradual linear slope.

    Parameters
    ----------
        start_val : float
            Starting value

        end_val : float
            Target value

        t_begin : int or array_like or :class:`~aesara.tensor.Variable`
            Time point (inbetween 0 and t_total) where start_val is placed

        t_end : int or array_like or :class:`~aesara.tensor.Variable`
            Time point (inbetween 0 and t_total) where end_val is placed

        t_total : int
            Total number of time points

    Returns
    -------
        : :class:`~aesara.tensor.Variable`
            vector of length t_total with the values of the parameterised f(t)
    """
    t = np.arange(t_total)

    return (
        at.clip((t - t_begin) / (t_end - t_begin), 0, 1) * (end_val - start_val)
        + start_val
    )
