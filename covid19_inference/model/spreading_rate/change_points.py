import logging
import pandas as pd
import pymc as pm
import numpy as np
import aesara.tensor as at
from .. import utility as ut
from ..model import modelcontext

log = logging.getLogger(__name__)


def _make_change_point_RVs(
    change_points_list,
    pr_median_lambda_0,
    pr_sigma_lambda_0=1,
    model=None,
    hierarchical=None,
    sigma_lambda_cp=None,
    sigma_lambda_week_cp=None,
    prefix_lambdas="",
    shape=None,
):
    """

    Parameters
    ----------
    priors_dict
    change_points_list
    model

    Returns
    -------

    TODO
    ----
        Documentation on this

        Add a way to name the changepoints
    """

    if shape is None:
        shape = model.shape_of_regions

    def hierarchical_mod():
        lambda_0_hc_L2_log, lambda_0_hc_L1_log = ut.hierarchical_normal(
            name_L1="lambda_0_hc_L1_log_",
            name_L2="lambda_0_hc_L2_log",
            name_sigma="sigma_lambda_0_hc_L1",
            pr_mean=np.log(pr_median_lambda_0),
            pr_sigma=pr_sigma_lambda_0,
            error_cauchy=False,
            shape=shape,
        )
        lambda_L1_log_list = []
        pm.Deterministic("lambda_0_hc_L2", at.exp(lambda_0_hc_L2_log))
        pm.Deterministic("lambda_0_hc_L1", at.exp(lambda_0_hc_L1_log))
        lambda_log_list.append(lambda_0_hc_L2_log)
        lambda_L1_log_list.append(lambda_0_hc_L1_log)

        # Create lambda_log list
        for i, cp in enumerate(change_points_list):
            if cp["relative_to_previous"]:
                if sigma_lambda_cp is None:
                    (
                        factor_lambda_cp_hc_L2_log,
                        factor_lambda_cp_hc_L1_log,
                    ) = ut.hierarchical_normal(
                        name_L1=f"factor_lambda_{i + 1}_hc_L1_log",
                        name_L2=f"factor_lambda_{i + 1}_hc_L2_log",
                        name_sigma=f"sigma_lambda_{i + 1}_hc_L1",
                        pr_mean=at.log(cp["pr_factor_to_previous"]),
                        pr_sigma=cp["pr_sigma_lambda"],
                        error_cauchy=False,
                        shape=shape,
                    )
                    lambda_cp_hc_L2_log = (
                        lambda_log_list[-1] + factor_lambda_cp_hc_L2_log
                    )
                    lambda_cp_hc_L1_log = (
                        lambda_L1_log_list[-1] + factor_lambda_cp_hc_L2_log
                    )
                else:
                    if sigma_lambda_week_cp is None:
                        raise RuntimeError("sigma_lambda_week_cp needs also to be set")
                    factor_lambda_week_log = (
                        pm.Normal(
                            name=f"diff_lambda_w_raw_{i+1}", mu=0, sigma=1.0, shape=()
                        )
                    ) * sigma_lambda_week_cp
                    lambda_cp_hc_L1_log = (
                        lambda_L1_log_list[-1] + factor_lambda_week_log
                    )
                    pr_mean_lambda = lambda_log_list[-1] + factor_lambda_week_log
                    lambda_cp_hc_L2_log = (
                        (
                            pm.Normal(
                                name=f"diff_lambda_cw_raw_{i + 1}",
                                mu=pr_mean_lambda,
                                sigma=1.0,
                                shape=shape,
                            )
                        )
                        - pr_mean_lambda
                    ) * sigma_lambda_cp + pr_mean_lambda

            else:
                pr_mean_lambda = np.log(cp["pr_median_lambda"])

                lambda_cp_hc_L2_log, lambda_cp_hc_L1_log = ut.hierarchical_normal(
                    name_L1=f"lambda_{i + 1}_hc_L1_log",
                    name_L2=f"lambda_{i + 1}_hc_L2_log",
                    name_sigma=f"sigma_lambda_{i + 1}_hc_L1",
                    pr_mean=pr_mean_lambda,
                    pr_sigma=cp["pr_sigma_lambda"],
                    error_cauchy=False,
                    shape=shape,
                )
            pm.Deterministic(f"lambda_{i + 1}_hc_L2", at.exp(lambda_cp_hc_L2_log))
            pm.Deterministic(f"lambda_{i + 1}_hc_L1", at.exp(lambda_cp_hc_L1_log))
            lambda_log_list.append(lambda_cp_hc_L2_log)
            lambda_L1_log_list.append(lambda_cp_hc_L1_log)

        # Create transient time list
        dt_before = model.sim_begin

        if hierarchical == "only_lambda":
            for i, cp in enumerate(change_points_list):
                dt_begin_transient = cp["pr_mean_date_transient"]
                if dt_before is not None and dt_before > dt_begin_transient:
                    raise RuntimeError(
                        "Dates of change points are not temporally ordered"
                    )

                prior_mean = (dt_begin_transient - model.sim_begin).days
                tr_time = pm.Normal(
                    name=f"{prefix_lambdas}transient_day_{i + 1}",
                    mu=prior_mean,
                    sigma=cp["pr_sigma_date_transient"],
                    shape=shape,
                )
                tr_time_list.append(tr_time)
                dt_before = dt_begin_transient

            # Create transient length list
            for i, cp in enumerate(change_points_list):
                tr_len_raw = pm.Normal(
                    name=f"{prefix_lambdas}transient_len_{i + 1}_raw_",
                    mu=cp["pr_median_transient_len"],
                    sigma=cp["pr_sigma_transient_len"],
                    shape=shape,
                )
                pm.Deterministic(
                    f"{prefix_lambdas}transient_len_{i + 1}",
                    at.softplus(tr_len_raw),
                )
                tr_len_list.append(at.softplus(tr_len_raw))
        else:
            for i, cp in enumerate(change_points_list):
                dt_begin_transient = cp["pr_mean_date_transient"]
                if dt_before is not None and dt_before > dt_begin_transient:
                    raise RuntimeError(
                        "Dates of change points are not temporally ordered"
                    )
                prior_mean = (dt_begin_transient - model.sim_begin).days
                tr_time_L2, _ = ut.hierarchical_normal(
                    name_L1=f"transient_day_{i + 1}_hc_L1",
                    name_L2=f"transient_day_{i + 1}_hc_L2",
                    name_sigma=f"sigma_transient_day_{i + 1}_L1",
                    pr_mean=prior_mean,
                    pr_sigma=cp["pr_sigma_date_transient"],
                    error_fact=1.0,
                    error_cauchy=False,
                    shape=shape,
                )
                tr_time_list.append(tr_time_L2)
                dt_before = dt_begin_transient

            # Create transient len list
            for i, cp in enumerate(change_points_list):
                # if model.sim_ndim == 1:
                tr_len_L2_raw, tr_len_L1_raw = ut.hierarchical_normal(
                    name_L1=f"transient_len_{i + 1}_hc_L1_raw",
                    name_L2=f"transient_len_{i + 1}_hc_L2_raw",
                    name_sigma=f"sigma_transient_len_{i + 1}",
                    pr_mean=cp["pr_median_transient_len"],
                    pr_sigma=cp["pr_sigma_transient_len"],
                    error_fact=1.0,
                    error_cauchy=False,
                    shape=shape,
                )
                if tr_len_L1_raw is not None:
                    pm.Deterministic(
                        f"transient_len_{i + 1}_hc_L1",
                        at.softplus(tr_len_L1_raw),
                    )
                    pm.Deterministic(
                        f"transient_len_{i + 1}_hc_L2",
                        at.softplus(tr_len_L2_raw),
                    )
                else:
                    pm.Deterministic(
                        f"transient_len_{i + 1}", at.softplus(tr_len_L2_raw)
                    )
                tr_len_list.append(at.softplus(tr_len_L2_raw))

    def non_hierarchical_mod():
        lambda_0_log = pm.Normal(
            name=f"{prefix_lambdas}lambda_0_log_",
            mu=np.log(pr_median_lambda_0),
            sigma=pr_sigma_lambda_0,
            shape=shape,
        )
        pm.Deterministic(f"{prefix_lambdas}lambda_0", at.exp(lambda_0_log))
        lambda_log_list.append(lambda_0_log)

        # Create lambda_log list
        for i, cp in enumerate(change_points_list):
            if cp["relative_to_previous"]:
                pr_mean_lambda = lambda_log_list[-1] + at.log(
                    cp["pr_factor_to_previous"]
                )
                if sigma_lambda_cp is not None:
                    lambda_cp_log = (
                        pm.Normal(
                            name=f"{prefix_lambdas}lambda_{i + 1}_log_",
                            mu=pr_mean_lambda,
                            sigma=1.0,
                            shape=shape,
                        )
                        - pr_mean_lambda
                    ) * sigma_lambda_cp + pr_mean_lambda

                else:
                    lambda_cp_log = pm.Normal(
                        name=f"{prefix_lambdas}lambda_{i + 1}_log_",
                        mu=pr_mean_lambda,
                        sigma=cp["pr_sigma_lambda"],
                        shape=shape,
                    )
            else:
                pr_mean_lambda = np.log(cp["pr_median_lambda"])
                lambda_cp_log = pm.Normal(
                    name=f"{prefix_lambdas}lambda_{i + 1}_log_",
                    mu=pr_mean_lambda,
                    sigma=cp["pr_sigma_lambda"],
                    shape=shape,
                )
            pm.Deterministic(f"{prefix_lambdas}lambda_{i + 1}", at.exp(lambda_cp_log))
            lambda_log_list.append(lambda_cp_log)

        # Create transient time list
        dt_before = model.sim_begin
        for i, cp in enumerate(change_points_list):
            dt_begin_transient = cp["pr_mean_date_transient"]
            if dt_before is not None and dt_before > dt_begin_transient:
                raise RuntimeError("Dates of change points are not temporally ordered")

            prior_mean = (dt_begin_transient - model.sim_begin).days
            tr_time = pm.Normal(
                name=f"{prefix_lambdas}transient_day_{i + 1}",
                mu=prior_mean,
                sigma=cp["pr_sigma_date_transient"],
                shape=shape,
            )
            tr_time_list.append(tr_time)
            dt_before = dt_begin_transient

        # Create transient length list
        for i, cp in enumerate(change_points_list):
            tr_len_raw = pm.Normal(
                name=f"{prefix_lambdas}transient_len_{i + 1}_raw_",
                mu=cp["pr_median_transient_len"],
                sigma=cp["pr_sigma_transient_len"],
                shape=shape,
            )
            pm.Deterministic(
                f"{prefix_lambdas}transient_len_{i + 1}",
                at.softplus(tr_len_raw),
            )
            tr_len_list.append(at.softplus(tr_len_raw))

    # ------------------------------------------------------------------------------ #
    # Start of function body
    # ------------------------------------------------------------------------------ #

    default_priors_change_points = dict(
        pr_median_lambda=pr_median_lambda_0,
        pr_sigma_lambda=pr_sigma_lambda_0,
        pr_sigma_date_transient=2,
        pr_median_transient_len=4,
        pr_sigma_transient_len=1,
        pr_mean_date_transient=None,
        relative_to_previous=False,
        pr_factor_to_previous=1,
    )

    for cp_priors in change_points_list:
        ut.set_missing_priors_with_default(cp_priors, default_priors_change_points)

    model = modelcontext(model)
    lambda_log_list = []
    tr_time_list = []
    tr_len_list = []

    if hierarchical is None:
        hierarchical = model.is_hierarchical
    if hierarchical:
        hierarchical_mod()
    else:
        non_hierarchical_mod()

    return lambda_log_list, tr_time_list, tr_len_list


def get_cps(data_begin, data_end, interval=7, offset=0, **priors_dict):
    """
    Generates and returns change point array.
    Parameters
    ----------
    data_begin : dateteime
        First date for possible changepoints
    data_end : datetime
        Last date for possible changepoints
    interval : int, optional
        Interval for the proposed cp in days. default:7
    offset : int, optional
        Offset for the first cp to data_begin
    """
    change_points = []
    count = interval - offset
    default_params = dict(
        pr_sigma_date_transient=1.5,
        pr_sigma_lambda=0.2,  # wiggle compared to previous point
        relative_to_previous=True,
        pr_factor_to_previous=1.0,
        pr_sigma_transient_len=1,
        pr_median_transient_len=4,
        pr_median_lambda=0.125,
    )
    ut.set_missing_priors_with_default(priors_dict, default_params)

    for day in pd.date_range(start=data_begin, end=data_end):
        if count / interval >= 1.0:
            # Add cp
            change_points.append(
                dict(  # one possible change point every sunday
                    pr_mean_date_transient=day, **priors_dict
                )
            )
            count = 1
        else:
            count = count + 1
    return change_points
