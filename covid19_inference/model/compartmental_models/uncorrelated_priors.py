import logging
import numpy as np
import pymc as pm

from aesara import scan
import aesara.tensor as at

from ..model import *
from .. import utility as ut

log = logging.getLogger(__name__)


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
    lambda_t_log : :class:`~aesara.tensor.TensorVariable`
        time series of the logarithm of the spreading rate, 1 or 2-dimensional. If 2-dimensional
        the first dimension is time.

    mu : :class:`pymc.Continuous`
        Distribution of the recovery rate :math:`\mu`. Defaults to
        :class:`pymc.LogNormal` with the arguments defined
        in ``mu_kwargs``. Can be 0 or 1-dimensional. If 1-dimensional, the dimension
        are the different regions.

    pr_median_delay : number
        TODO

    n_data_points_used : int, optional
        TODO

    pr_sigma_I_begin : int, optional
        TODO

    Other Parameters
    ----------------
    name_I_begin : str, optional
        TODO

    name_I_begin_ratio_log : str, optional
        TODO

    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    Returns
    ------------------
    I_begin: :class:`~aesara.tensor.TensorVariable`

    """
    log.info("Compartmental Model (Uncorrelated prior I)")
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

    # I_begin = pm.LogNormal(
    #   name_I_begin_ratio_log, mu=at.log(I0_ref), sigma=2.5, shape=num_regions
    # )
    if name_I_begin is not None:
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
    Builds the prior for E begin  by solving the SEIR differential from the first
    data backwards. This decorrelates the I_begin from the lambda_t at the
    beginning, allowing a more efficient sampling. The example_one_bundesland runs
    about 30\% faster with this prior, instead of a HalfCauchy.

    Parameters
    ----------
    pr_median_delay : TYPE
        Description
    name_E_begin : str, optional
        TODO
    name_E_begin_ratio_log : str, optional
        TODO
    pr_sigma_E_begin : int, optional
        TODO
    n_data_points_used : int, optional
        TODO
    len_time : int, optional
        TODO

    Other Parameters
    ----------------
    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    Returns
    ------------------
    E_begin: :class:`~aesara.tensor.TensorVariable`

    """
    log.info("Compartmental Model (Uncorrelated prior E)")
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
