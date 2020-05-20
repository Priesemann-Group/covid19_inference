# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-05-19 11:58:22
# @Last Modified: 2020-05-20 10:53:03
# ------------------------------------------------------------------------------ #

import logging

import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm

log = logging.getLogger(__name__)

from .model import *

# week_modulation.py
def week_modulation(
    new_cases_raw,
    week_modulation_type="abs_sine",
    pr_mean_weekend_factor=0.3,
    pr_sigma_weekend_factor=0.5,
    week_end_days=(6, 7),
    model=None,
    save_in_trace=True,
):
    r"""
    Adds a weekly modulation of the number of new cases:

    .. math::
        \text{new\_cases} &= \text{new\_cases\_raw} \cdot (1-f(t))\,, \qquad\text{with}\\
        f(t) &= f_w \cdot \left(1 - \left|\sin\left(\frac{\pi}{7} t- \frac{1}{2}\Phi_w\right)\right| \right),

    if ``week_modulation_type`` is ``"abs_sine"`` (the default). If ``week_modulation_type`` is ``"step"``, the
    new cases are simply multiplied by the weekend factor on the days set by ``week_end_days``

    The weekend factor :math:`f_w` follows a Lognormal distribution with
    median ``pr_mean_weekend_factor`` and sigma ``pr_sigma_weekend_factor``. It is hierarchically constructed if
    the input is two-dimensional by the function :func:`hierarchical_normal` with default arguments.

    The offset from Sunday :math:`\Phi_w` follows a flat :class:`~pymc3.distributions.continuous.VonMises` distribution
    and is the same for all regions.

    Parameters
    ----------

    new_cases_raw : :class:`~theano.tensor.TensorVariable`
        The input array, can be one- or two-dimensional
    week_modulation_type : str
        The type of modulation, accepts ``"step"`` or  ``"abs_sine`` (the default).
    pr_mean_weekend_factor : float
        Sets the prior mean of the factor :math:`f_w` by which weekends are counted.
    pr_sigma_weekend_factor : float
        Sets the prior sigma of the factor :math:`f_w` by which weekends are counted.
    week_end_days : tuple of ints
        The days counted as weekend if ``week_modulation_type`` is ``"step"``
    model : :class:`Cov19Model`
        if none, it is retrieved from the context
    save_in_trace : bool
        If True (default) the new_cases are saved in the trace.

    Returns
    -------

    new_cases : :class:`~theano.tensor.TensorVariable`

    """

    def step_modulation():
        """
        Helper function for the step modulation

        Returns
        -------
        modulation
        """
        modulation = np.zeros(shape_modulation[0])
        for i in range(shape_modulation[0]):
            date_curr = model.data_begin + datetime.timedelta(days=i)
            if date_curr.isoweekday() in week_end_days:
                modulation[i] = 1
        return modulation

    def abs_sine_modulation():
        """
        Helper function for the absolute sin modulation

        Returns
        -------
        modulation
        """
        offset_rad = pm.VonMises("offset_modulation_rad", mu=0, kappa=0.01)
        offset = pm.Deterministic("offset_modulation", offset_rad / (2 * np.pi) * 7)
        t = np.arange(shape_modulation[0]) - model.data_begin.weekday()  # Sunday @ zero
        modulation = 1 - tt.abs_(tt.sin(t / 7 * np.pi + offset_rad / 2))
        return modulation

    #Create our model context
    model = modelcontext(model)

    #Get the shape of the modulation from the shape of our simulation
    shape_modulation = list(model.sim_shape)
    shape_modulation[0] -= model.diff_data_sim
        
    if not model.is_hierarchical:
        weekend_factor_log = pm.Normal(
            name="weekend_factor_log",
            mu=tt.log(pr_mean_weekend_factor),
            sigma=pr_sigma_weekend_factor
            )
        week_end_factor = tt.exp(weekend_factor_log)
        pm.Deterministic("weekend_factor", week_end_factor)

    else: #hierarchical  
        week_end_factor_L2_log, week_end_factor_L1_log = hierarchical_normal(
            name_L1 ="weekend_factor_hc_L1_log",
            name_L2 ="weekend_factor_hc_L1_log",
            name_sigma="sigma_weekend_factor",
            pr_mean=tt.log(pr_mean_weekend_factor),
            pr_sigma=pr_sigma_weekend_factor,
            )

        # We do that so we can use it later (same name as non hierarchical)
        week_end_factor = tt.exp(week_end_factor_L2_log)

        pm.Deterministic("weekend_factor_L2", weekend_factor)
        pm.Deterministic("weekend_factor_L1", tt.exp(week_end_factor_L1_log))

    # Different modulation types
    modulation = step_modulation() if week_modulation_type == "step" else 0
    modulation = abs_sine_modulation() if week_modulation_type == "abs_sine" else 0

    if model.is_hierarchical:
        modulation = tt.shape_padaxis(modulation, axis=-1)
        
    multiplication_vec = tt.abs_(
        np.ones(shape_modulation) - week_end_factor * modulation
    )


    new_cases_inferred_eff = new_cases_raw * multiplication_vec

    if save_in_trace:
        pm.Deterministic("new_cases", new_cases_inferred_eff)

    return new_cases_inferred_eff
