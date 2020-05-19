# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-05-19 11:58:22
# @Last Modified: 2020-05-19 12:02:34
# ------------------------------------------------------------------------------ #

import logging

import theano
import theano.tensor as tt
import numpy as np

log = logging.getLogger(__name__)

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
    model = modelcontext(model)
    shape_modulation = list(model.sim_shape)
    shape_modulation[0] -= model.diff_data_sim

    len_L2 = () if model.sim_ndim == 1 else model.sim_shape[1]

    week_end_factor_L2_log, week_end_factor_L1_log = hierarchical_normal(
        "weekend_factor_log",
        "sigma_weekend_factor",
        pr_mean=tt.log(pr_mean_weekend_factor),
        pr_sigma=pr_sigma_weekend_factor,
        len_L2=len_L2,
    )

    week_end_factor_L2 = tt.exp(week_end_factor_L2_log)

    if model.sim_ndim == 1:
        pm.Deterministic("weekend_factor", week_end_factor_L2)
    elif model.sim_ndim == 2:
        week_end_factor_L1 = tt.exp(week_end_factor_L1_log)
        pm.Deterministic("weekend_factor_L2", week_end_factor_L2)
        pm.Deterministic("weekend_factor_L1", week_end_factor_L1)

    if week_modulation_type == "step":
        modulation = np.zeros(shape_modulation[0])
        for i in range(shape_modulation[0]):
            date_curr = model.data_begin + datetime.timedelta(days=i)
            if date_curr.isoweekday() in week_end_days:
                modulation[i] = 1
    elif week_modulation_type == "abs_sine":
        offset_rad = pm.VonMises("offset_modulation_rad", mu=0, kappa=0.01)
        offset = pm.Deterministic("offset_modulation", offset_rad / (2 * np.pi) * 7)
        t = np.arange(shape_modulation[0]) - model.data_begin.weekday()  # Sunday @ zero
        modulation = 1 - tt.abs_(tt.sin(t / 7 * np.pi + offset_rad / 2))

    if model.sim_ndim == 2:
        modulation = tt.shape_padaxis(modulation, axis=-1)

    multiplication_vec = tt.abs_(
        np.ones(shape_modulation) - week_end_factor_L2 * modulation
    )
    new_cases_inferred_eff = new_cases_raw * multiplication_vec
    if save_in_trace:
        pm.Deterministic("new_cases", new_cases_inferred_eff)
    return new_cases_inferred_eff
