# ------------------------------------------------------------------------------ #
# Apply a weekly modulation to the reported cases. Less reports on the weekend
# ------------------------------------------------------------------------------ #

import logging
import aesara
import numpy as np
from datetime import timedelta
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
        "by_weekday": by_weekday_modulation,
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
    weekend_days=(6, 7),
    # weekend_factor,
    weekend_factor=None,
    weekend_factor_kwargs={
        "name": "weekend_factor",
        "mu": np.log(0.3),
        "sigma": 0.5,
    },
    # other
    model=None,
):
    r""" Adds a step modulation to the cases to account for weekend not
    being reported or being reported less.
    .. math::
        \text{cases} &= \text{cases_raw} \cdot (1-f(t))\,, \qquad\text{with}\\
        f(t) &= \begin{cases}
            1 & \text{if } t \text{ is a weekend day}\\
            0 & \text{otherwise}
        \end{cases}
    
    By default, weekend factor :math:`f_w` follows a LogNormal distribution with
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
    weekend_factor : None or :class:`pymc.Continuous`
        The weekend factor :math:`f_w` can be passed as a PyMC distribution. If ``None`` it is
        constructed from the parameters ``weekend_factor_prior_mu`` and ``weekend_factor_prior_sigma`` using 
        a Lognormal distribution. If the input is two-dimensional, the distribution is hierarchically constructed
        using the function :func:`hierarchical_normal` with default arguments.
    weekend_factor_kwargs : dict
        Keyword arguments passed to the pymc distribution of ``weekend_factor`` if it is constructed.
        See :class:`pymc.Normal` for available arguments. Default is ``{"name":"weekend_factor", "mu": log(0.3), "sigma": 0.5}``.
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
        # Create LogNormal distribution f_w
        weekend_factor_name = weekend_factor_kwargs.pop("name", "weekend_factor")
        if not model.is_hierarchical:
            weekend_factor_log = pm.Normal(
                name=weekend_factor_name + "_log",
                **weekend_factor_kwargs,
            )
            weekend_factor = pm.Deterministic(
                weekend_factor_name, at.exp(weekend_factor_log)
            )
        else:  # hierarchical
            weekend_factor_L2_log, weekend_factor_L1_log = ut.hierarchical_normal(
                name_L1=weekend_factor_name + "_hc_L1_log",
                name_L2=weekend_factor_name + "_hc_L2_log",
                name_sigma="sigma_" + weekend_factor_name,
                pr_mean=weekend_factor_kwargs["mu"],
                pr_sigma=weekend_factor_kwargs["sigma"],
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
        date_curr = model.sim_begin + timedelta(days=i)
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
    weekend_factor_kwargs={
        "name": "weekend_factor",
        "mu": np.log(0.3),
        "sigma": 0.5,
    },
    # offset_modulation,
    offset_modulation=None,
    offset_modulation_kwargs={
        "name": "offset_modulation",
        "mu": 0,
        "kappa": 0.01,
    },
    # other
    model=None,
):
    r"""Adds weekly modulation to the cases, following an absolute sine function.
    .. math::
        \text{cases} &= \text{cases_raw} \cdot (1-f(t))\,, \qquad\text{with}\\
        f(t) &= f_w \cdot \left(1 - \left|\sin\left(\frac{\pi}{7} t- \frac{1}{2}\Phi_w\right)\right| \right),
    By default, weekend factor :math:`f_w` follows a Lognormal distribution with
    median ``weekend_factor_prior_mu`` and sigma ``weekend_factor_prior_sigma``. It is hierarchically
    constructed if the input is two-dimensional by the function :func:`hierarchical_normal` with default arguments.
    The offset from Sunday :math:`\Phi_w` follows a flat :class:`pymc.VonMises`
    distribution and is the same for all regions.
    This method was used in the paper `Dehning et al. 2020`_. It might not be
    the best choice and depends on the reporting behavior of the region.
    .. _Dehning et al. 2020: https://www.science.org/doi/10.1126/science.abb9789
    Parameters
    ----------
    cases_raw : :class:`~aesara.tensor.TensorVariable`
        The input array of daily new cases, can be one- or two-dimensional. First dimension
        has to be the time dimension.
    weekend_factor : None or :class:`pymc.Continuous`
        The weekend factor :math:`f_w` can be passed as a Pymc distribution. If ``None`` it is
        constructed from the parameters ``weekend_factor_prior_mu`` and ``weekend_factor_prior_sigma`` using 
        a Lognormal distribution. If the input is two-dimensional, the distribution is hierarchically constructed
        using the function :func:`hierarchical_normal` with default arguments.
    weekend_factor_kwargs : dict
        Keyword arguments passed to the pymc distribution of ``weekend_factor`` if it is constructed.
        See :class:`pymc.Normal` for available arguments. Default is ``{"name":"weekend_factor", "mu": log(0.3), "sigma": 0.5}``.
    offset_modulation : None or :class:`pymc.Continuous`
        The offset from Sunday :math:`\Phi_w` can be passed as a PyMC distribution. If ``None`` it is a flat VonMises distribution (mu=0, kappa=0.01).
    offset_modulation_kwargs : dict
        Keyword arguments passed to the pymc distribution of ``offset_modulation`` if it is constructed.
        See :class:`pymc.VonMises` for available arguments. Default is ``{"name":"offset_modulation", "mu": 0, "kappa": 0.01}``.
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
        # Create LogNormal distribution f_w
        weekend_factor_name = weekend_factor_kwargs.pop("name", "weekend_factor")
        
        if not model.is_hierarchical:
            weekend_factor_log = pm.Normal(
                name=weekend_factor_name + "_log", **weekend_factor_kwargs
            )
            weekend_factor = pm.Deterministic(
                weekend_factor_name, at.exp(weekend_factor_log)
            )
        else:  # hierarchical
            weekend_factor_L2_log, weekend_factor_L1_log = ut.hierarchical_normal(
                name_L1=weekend_factor_name + "_hc_L1_log",
                name_L2=weekend_factor_name + "_hc_L2_log",
                name_sigma="sigma_" + weekend_factor_name,
                pr_mean=weekend_factor_kwargs["mu"],
                pr_sigma=weekend_factor_kwargs["sigma"],
            )
            weekend_factor = at.exp(weekend_factor_L2_log)
            pm.Deterministic(
                weekend_factor_name + "_hc_L1", at.exp(weekend_factor_L1_log)
            )
            pm.Deterministic(
                weekend_factor_name + "_hc_L2", at.exp(weekend_factor_L2_log)
            )

    # Check if offset_modulation is passed as a distribution
    if offset_modulation is None:
        offset_modulation_name = offset_modulation_kwargs.pop(
            "name", "offset_modulation"
        )
        offset_modulation = pm.VonMises(
            offset_modulation_name + "_rad", **offset_modulation_kwargs
        )
        pm.Deterministic(offset_modulation_name, offset_modulation / (2 * np.pi) * 7)

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


def by_weekday_modulation(cases, model=None):
    """Delayed cases by weekday.
    Only works for one region atm.
    Parameters
    TODO
    """

    log.info("Week modulation (by weekday)")
    model = modelcontext(model)

    r_base_high = pm.Normal("fraction_delayed_weekend_raw", mu=-3, sigma=2)
    r_base_low = pm.Normal("fraction_delayed_week_raw", mu=-5, sigma=1)
    sigma_r = pm.HalfNormal("sigma_r", sigma=1)

    delta_r = (pm.Normal("delta_fraction_delayed", mu=0, sigma=1, shape=7)) * sigma_r
    e = pm.HalfCauchy("error_fraction", beta=0.2)

    r_base = at.stack(
        [
            r_base_high,
            r_base_low,
            r_base_low,
            r_base_low,
            r_base_high,
            r_base_high,
            r_base_high,
        ]  # Monday @ zero
    )
    r_week = r_base + delta_r

    r_transformed_week = at.nnet.sigm.sigmoid(r_week)
    pm.Deterministic("mean_fraction_delayed_by_weekday", r_transformed_week)

    t = np.arange(model.sim_shape[0]) + model.sim_begin.weekday()  # Monday @ zero

    week_matrix = np.zeros((model.sim_shape[0], 7), dtype="float")
    week_matrix[np.stack([t] * 7, axis=1) % 7 == np.arange(7)] = 1.0

    r_t = at.dot(week_matrix, r_transformed_week)

    fraction = pm.Beta(
        "fraction_delayed", alpha=r_t / e, beta=(1 - r_t) / e, shape=model.sim_shape[0]
    )

    def loop_delay_by_weekday(C_t, fraction_t, C_tm1, fraction_tm1):
        new_cases = (1.0 - fraction_t) * (C_t + fraction_tm1 * C_tm1)
        return new_cases, fraction_t

    (cases_modulated, _), _ = scan(
        fn=loop_delay_by_weekday,
        sequences=[cases, fraction],
        outputs_info=[
            cases[0],
            fraction[0],
        ],
        strict=True,
        n_steps=cases.shape[0],
        non_sequences=[],
    )

    pm.Deterministic("delayed_cases_by_weekday", cases_modulated)
    return cases_modulated
