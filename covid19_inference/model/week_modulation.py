# ------------------------------------------------------------------------------ #
# Apply a weekly modulation to the reported cases. Less reports on the weekend
# ------------------------------------------------------------------------------ #

import logging
import numpy as np

import pymc as pm
from aesara import scan
import aesara.tensor as at


from .model import *
from . import utility as ut

log = logging.getLogger(__name__)


def week_modulation(
    cases,
    week_modulation_type="abs_sine",
    model=None,
    **kwargs,
):
    r"""
    This is a high level function to apply a weekly modulation to the reported cases.

    The type of the modulation is set by ``week_modulation_type``. The following types are available:

    * ``abs_sine``: The absolute value of a sine function is applied to the cases. This is the default. See :func:`abs_sine_modulation` for more details.
    * ``step``: A step function is applied to the cases. See :func:`step_modulation` for more details.

    All function will use the default parameters but you
    can supply parameters by passing them as keyword arguments.

    Parameters
    ----------
    cases_raw : :class:`~aesara.tensor.TensorVariable`
        The input array of daily new cases, can be one- or two-dimensional.
    week_modulation_type : str
        The type of modulation, accepts ``"step"`` or  ``"abs_sine`` (the default).
    model : :class:`Cov19Model`
        If none, it is retrieved from the context
    kwargs : dict
        Additional keyword arguments are passed to the respective lower level functions.

    Returns
    -------
    cases : :class:`~aesara.tensor.TensorVariable`
        The new cases with modulation applied.

    """

    # Switch for the different types of modulation
    types = {
        "abs_sine": abs_sine_modulation,
        "step": step_modulation,
    }

    if week_modulation_type not in types:
        raise ValueError(
            f"week_modulation_type must be one of {types.keys()}, but is {week_modulation_type}"
        )

    # Use the function
    cases = types[week_modulation_type](cases, model=model, **kwargs)

    return cases


def step_modulation(
    cases_raw,
    weekend_days=(5, 6),
    # weekend_factor,
    weekend_factor=None,
    weekend_factor_name="weekend_factor",
    weekend_factor_prior_mu=0.3,
    weekend_factor_prior_sigma=0.5,
    # other
    model=None,
):
    r""" Adds a step modulation to the cases to account for weekend not
    being reported or being reported less.

    .. math::
        \text{cases} &= \text{cases\_raw} \cdot (1-f(t))\,, \qquad\text{with}\\
        f(t) &= \begin{cases}
            1 & \text{if } t \text{ is a weekend day}\\
            0 & \text{otherwise}
        \end{cases}
    
    By default, weekend factor :math:`f_w` follows a Lognormal distribution with
    median ``weekend_factor_prior_mu`` and sigma ``weekend_factor_prior_sigma``. It is hierarchically
    constructed if the input is two-dimensional by the function :func:`hierarchical_normal` with default arguments.

    Weekends are defined to be Saturday and Sunday, but can be changed with
    the ``weekend_days`` parameter.

    Parameters
    ----------

    cases_raw : :class:`~aesara.tensor.TensorVariable`
        The input array of daily new cases, can be one- or two-dimensional. First dimension
        has to be the time dimension.

    weekend_days : tuple of ints
        The days counted as weekend (isoweekday: 1 is Monday, 7 is Sunday). Default is Saturday and Sunday (6,7).

    weekend_factor : None or :class:`~pm.distributions.Continuous`
        The weekend factor :math:`f_w` can be passed as a PyMC3 distribution. If ``None`` it is
        constructed from the parameters ``weekend_factor_prior_mu`` and ``weekend_factor_prior_sigma`` using 
        a Lognormal distribution. If the input is two-dimensional, the distribution is hierarchically constructed
        using the function :func:`hierarchical_normal` with default arguments.

    weekend_factor_prior_mu : float
        The prior mean of the weekend factor :math:`f_w` if it is constructed from a Lognormal distribution.
        Ignored if ``weekend_factor`` is not ``None``.
    
    weekend_factor_prior_sigma : float
        The prior sigma of the weekend factor :math:`f_w` if it is constructed from a Lognormal distribution.
        Ignored if ``weekend_factor`` is not ``None``.
    
    weekend_factor_name : str
        The name of the weekend factor :math:`f_w` if it is constructed from a Lognormal distribution.

    Returns
    -------
    cases : :class:`~aesara.tensor.TensorVariable`
        The new cases with modulation applied.
    """

    log.info("Week modulation (step)")
    model = modelcontext(None)
    # Get the shape of the modulation from the shape of our simulation
    shape_modulation = list(model.sim_shape)

    # Check if weekend_factor is passed as a distribution
    if weekend_factor is None:
        if (
            weekend_factor_prior_mu is None
            or weekend_factor_prior_sigma is None
            or weekend_factor_name is None
        ):
            raise ValueError(
                "If weekend_factor is not passed as a distribution, weekend_factor_prior_mu, weekend_factor_prior_sigma and weekend_factor_name have to be set."
            )

        # Create LogNormal distribution f_w
        if not model.is_hierarchical:
            weekend_factor_log = pm.Normal(
                name=weekend_factor_name + "_log",
                mu=at.log(weekend_factor_prior_mu),
                sigma=weekend_factor_prior_sigma,
            )
            weekend_factor = at.exp(weekend_factor_log)
            pm.Deterministic(weekend_factor_name, weekend_factor)
        else:  # hierarchical
            weekend_factor_L2_log, weekend_factor_L1_log = ut.hierarchical_normal(
                name_L1=weekend_factor_name + "_hc_L1_log",
                name_L2=weekend_factor_name + "_hc_L2_log",
                name_sigma="sigma_" + weekend_factor_name,
                pr_mean=at.log(weekend_factor_prior_mu),
                pr_sigma=weekend_factor_prior_sigma,
            )
            weekend_factor = at.exp(weekend_factor_L2_log)
            pm.Deterministic(
                weekend_factor_name + "_hc_L1", at.exp(weekend_factor_L1_log)
            )
            pm.Deterministic(
                weekend_factor_name + "_hc_L2", at.exp(weekend_factor_L2_log)
            )
    elif not isinstance(weekend_factor, pm.Distribution):
        raise ValueError("weekend_factor has to be a PyMC distribution.")

        # Modulation
    modulation = np.zeros(shape_modulation[0])
    for i in range(shape_modulation[0]):
        date_curr = model.sim_begin + datetime.timedelta(days=i)
        if date_curr.isoweekday() in weekend_days:
            modulation[i] = 1

    modulation = at.as_tensor_variable(modulation)

    # Pad modulation to the shape of the simulation
    if len(shape_modulation) > 1:
        for i in range(len(shape_modulation) - 1):
            modulation = at.shape_padaxis(modulation, axis=-1)

    # Apply modulation
    cases = cases_raw * at.abs(np.ones(model.sim_shape) - weekend_factor * modulation)

    return cases


def abs_sine_modulation(
    cases_raw,
    # weekend_factor,
    weekend_factor=None,
    weekend_factor_name="weekend_factor",
    weekend_factor_prior_mu=0.3,
    weekend_factor_prior_sigma=0.5,
    # offset_modulation,
    offset_modulation=None,
    offset_modulation_name="offset_modulation",
    # other
    model=None,
):
    r"""Adds weekly modulation to the cases, following an absolute sine function.

    .. math::
        \text{cases} &= \text{cases\_raw} \cdot (1-f(t))\,, \qquad\text{with}\\
        f(t) &= f_w \cdot \left(1 - \left|\sin\left(\frac{\pi}{7} t- \frac{1}{2}\Phi_w\right)\right| \right),


    By default, weekend factor :math:`f_w` follows a Lognormal distribution with
    median ``weekend_factor_prior_mu`` and sigma ``weekend_factor_prior_sigma``. It is hierarchically
    constructed if the input is two-dimensional by the function :func:`hierarchical_normal` with default arguments.

    The offset from Sunday :math:`\Phi_w` follows a flat :class:`~pymc3.distributions.continuous.VonMises`
    distribution and is the same for all regions.

    This method was used in the paper `Dehning et al. 2020`_. It might not be
    the best choice and depends on the reporting behavior of the region.

    .. _Dehning et al. 2020: https://www.science.org/doi/10.1126/science.abb9789

    Parameters
    ----------
    cases_raw : :class:`~aesara.tensor.TensorVariable`
        The input array of daily new cases, can be one- or two-dimensional. First dimension
        has to be the time dimension.

    weekend_factor : None or :class:`~pm.distributions.Continuous`
        The weekend factor :math:`f_w` can be passed as a PyMC3 distribution. If ``None`` it is
        constructed from the parameters ``weekend_factor_prior_mu`` and ``weekend_factor_prior_sigma`` using 
        a Lognormal distribution. If the input is two-dimensional, the distribution is hierarchically constructed
        using the function :func:`hierarchical_normal` with default arguments.

    weekend_factor_prior_mu : float
        The prior mean of the weekend factor :math:`f_w` if it is constructed from a Lognormal distribution.
        Ignored if ``weekend_factor`` is not ``None``.
    
    weekend_factor_prior_sigma : float
        The prior sigma of the weekend factor :math:`f_w` if it is constructed from a Lognormal distribution.
        Ignored if ``weekend_factor`` is not ``None``.
    
    weekend_factor_name : str
        The name of the weekend factor :math:`f_w` if it is constructed from a Lognormal distribution.

    offset_modulation : None or :class:`~pm.distributions.Continuous`
        The offset from Sunday :math:`\Phi_w` can be passed as a PyMC3 distribution. If ``None`` it is a flat VonMises distribution (mu=0, kappa=0.01).

    offset_modulation_name : str
        The name of the offset from Sunday :math:`\Phi_w` if it is constructed from a VonMises distribution.

    Returns
    -------
    cases : :class:`~aesara.tensor.TensorVariable`
        The new cases with modulation applied.
    """

    log.info("Week modulation (absolute sine)")
    model = modelcontext(model)
    # Get the shape of the modulation from the shape of our simulation
    shape_modulation = list(model.sim_shape)

    # Check if weekend_factor is passed as a distribution
    if weekend_factor is None:
        if (
            weekend_factor_prior_mu is None
            or weekend_factor_prior_sigma is None
            or weekend_factor_name is None
        ):
            raise ValueError(
                "If weekend_factor is not passed as a distribution, weekend_factor_prior_mu, weekend_factor_prior_sigma and weekend_factor_name have to be set."
            )

        # Create LogNormal distribution f_w
        if not model.is_hierarchical:
            weekend_factor_log = pm.Normal(
                name=weekend_factor_name + "_log",
                mu=at.log(weekend_factor_prior_mu),
                sigma=weekend_factor_prior_sigma,
            )
            weekend_factor = at.exp(weekend_factor_log)
            pm.Deterministic(weekend_factor_name, weekend_factor)
        else:  # hierarchical
            weekend_factor_L2_log, weekend_factor_L1_log = ut.hierarchical_normal(
                name_L1=weekend_factor_name + "_hc_L1_log",
                name_L2=weekend_factor_name + "_hc_L2_log",
                name_sigma="sigma_" + weekend_factor_name,
                pr_mean=at.log(weekend_factor_prior_mu),
                pr_sigma=weekend_factor_prior_sigma,
            )
            weekend_factor = at.exp(weekend_factor_L2_log)
            pm.Deterministic(
                weekend_factor_name + "_hc_L1", at.exp(weekend_factor_L1_log)
            )
            pm.Deterministic(
                weekend_factor_name + "_hc_L2", at.exp(weekend_factor_L2_log)
            )
    elif not isinstance(weekend_factor, pm.Distribution):
        raise ValueError("weekend_factor has to be a PyMC distribution.")

    # Check if offset_modulation is passed as a distribution
    if offset_modulation is None:
        offset_modulation = pm.VonMises(
            offset_modulation_name + "_rad", mu=0, kappa=0.01
        )
        pm.Deterministic(offset_modulation_name, offset_modulation / (2 * np.pi) * 7)
    elif not isinstance(offset_modulation, pm.Distribution):
        raise ValueError("offset_modulation has to be a PyMC distribution.")

    # Modulation
    t = at.arange(shape_modulation[0]) - model.sim_begin.weekday()  # Sunday is 0
    modulation = 1 - at.abs(at.sin(t / 7 * np.pi + offset_modulation / 2))

    # Pad modulation to the shape of the simulation
    if len(shape_modulation) > 1:
        for i in range(len(shape_modulation) - 1):
            modulation = at.shape_padaxis(modulation, axis=-1)

    # Apply modulation
    cases = cases_raw * at.abs(np.ones(model.sim_shape) - weekend_factor * modulation)

    return cases
