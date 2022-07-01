# ------------------------------------------------------------------------------ #
# This file implements the change points in the spreading rate
# ------------------------------------------------------------------------------ #

import logging
import pymc as pm
import aesara.tensor as at

import scipy.special
from .model import *
from . import utility as ut

log = logging.getLogger(__name__)


def R_t_log_with_longtailed_dist(
    R_0_log, dist=None, model=None, name_R_t=None, rho_scale_prior=0.1, **dist_priors
):
    r"""
    Builds a time dependent reproduction number :math:`R_t` by using a long tailed distribution.

    .. math::

        log(R_t) &= log(R_0) + cumsum(  \cdot \Delta rho))
        s &\sim \text{HalfCauchy}(\text{rho_scale_prior})

    The parameter :math:`\Delta rho` is dependent on the given distribution. Possible values are
    ``studentT``,``weibull``,``cauchy``. In theory you can also pass your another function of type
    :class:`pm.distributions.Continuous`.


    Parameters
    ----------
    R_0_log : number
        Initial value of the reproduction number in log transform
    dist : string or :class:`pm.distributions.Continuous`
        Distribution to use. Possible values are ``studentT``,``weibull``,``cauchy``.
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

    # Create responding R_t pymc3 variable with given name (from parameters)
    if name_R_t is not None:
        pm.Deterministic(name_R_t, at.exp(R_t_log))

    return R_t_log


def lambda_t_with_sigmoids(
    change_points_list,
    pr_median_lambda_0,
    pr_sigma_lambda_0=0.5,
    model=None,
    name_lambda_t="lambda_t",
    hierarchical=None,
    sigma_lambda_cp=None,
    sigma_lambda_week_cp=None,
    prefix_lambdas="",
    shape=None,
):
    r"""
    Builds a time dependent spreading rate :math:`\lambda_t` with change points. The change points are marked by a transient with a sigmoidal shape.

    TODO
    ----
    Add a bit more detailed documentation.

    Parameters
    ----------
    change_points_list
    pr_median_lambda_0
    pr_sigma_lambda_0
    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    Returns
    -------
    lambda_t_log
    """
    log.info("Lambda_t with sigmoids")
    # Get our default mode context
    model = modelcontext(model)
    hierarchical = (
        model.is_hierarchical if model.is_hierarchical is None else hierarchical
    )
    # ?Get change points random variable?
    if shape is None:
        shape = model.shape_of_regions
    if isinstance(shape, int):
        shape = (shape,)

    lambda_log_list, tr_time_list, tr_len_list = _make_change_point_RVs(
        change_points_list,
        pr_median_lambda_0,
        pr_sigma_lambda_0,
        model=model,
        hierarchical=hierarchical,
        sigma_lambda_cp=sigma_lambda_cp,
        sigma_lambda_week_cp=sigma_lambda_week_cp,
        prefix_lambdas=prefix_lambdas,
        shape=shape,
    )

    # Build the time-dependent spreading rate
    lambda_log_t_list = [lambda_log_list[0] * at.ones((model.sim_len,) + shape)]
    lambda_before = lambda_log_list[0]

    # Loop over all lambda values and there corresponding transient values
    for tr_time, tr_len, lambda_after in zip(
        tr_time_list, tr_len_list, lambda_log_list[1:]
    ):
        # Create the right shape for the time array
        t = np.arange(model.sim_shape[0])

        # If the model is hierarchical repeatly add the t array to itself to match the shape
        if shape:
            t = np.repeat(t[:, None], shape, axis=-1)

        # Applies standart sigmoid nonlinearity
        lambda_t = at.nnet.basic.sigmoid((t - tr_time) / tr_len * 4) * (
            lambda_after - lambda_before
        )  # tr_len*4 because the derivative of the sigmoid at zero is 1/4, we want to set it to 1/tr_len

        lambda_before = lambda_after
        lambda_log_t_list.append(lambda_t)

    # Sum up all lambda values from the list
    for i, lambda_t in enumerate(lambda_log_t_list):
        pm.Deterministic(f"{prefix_lambdas}lambda_t_part_{i}", lambda_t)

    lambda_t_log = sum(lambda_log_t_list)

    # Create responding lambda_t pymc3 variable with given name (from parameters)
    if name_lambda_t is not None:
        pm.Deterministic(name_lambda_t, at.exp(lambda_t_log))

    return lambda_t_log


def lambda_t_with_linear_interp(
    change_points_list,
    pr_median_lambda_0,
    pr_sigma_lambda_0=0.5,
    model=None,
    name_lambda_t="lambda_t",
):
    """
    Parameters
    ----------
    change_points_list
    pr_median_lambda_0
    pr_sigma_lambda_0
    model : :class:`Cov19Model`
        if none, it is retrieved from the context

    Returns
    -------
    lambda_t_log

    TODO
    ----
    Documentation on this
    """
    log.info("Lambda_t linear in lin-space")
    # Get our default mode context
    model = modelcontext(model)

    # ?Get change points random variable?
    lambda_log_list, tr_time_list, tr_len_list = _make_change_point_RVs(
        change_points_list, pr_median_lambda_0, pr_sigma_lambda_0, model=model
    )

    # Build the time-dependent spreading rate
    lambda_t_list = [
        at.exp(lambda_log_list[0]) * at.ones(model.sim_shape)
    ]  # model.sim_shape = (time, state)
    lambda_log_before = lambda_log_list[0]

    # Loop over all lambda values and there corresponding transient values
    for tr_time, tr_len, lambda_log_after in zip(
        tr_time_list, tr_len_list, lambda_log_list[1:]
    ):
        # Create the right shape for the time array
        t = np.arange(model.sim_shape[0])

        # If the model is hierarchical repeatly add the t array to itself to match the shape
        if model.is_hierarchical:
            t = np.repeat(t[:, None], model.sim_shape[1], axis=-1)

        step_t = at.clip((t - tr_time + tr_len / 2) / tr_len, 0, 1)

        # create the time series of lambda
        lambda_t = step_t * (at.exp(lambda_log_after) - at.exp(lambda_log_before))

        lambda_log_before = lambda_log_after
        lambda_t_list.append(lambda_t)

    # Sum up all lambda values from the list
    lambda_t_log = at.log(sum(lambda_t_list))

    # Create responding lambda_t pymc3 variable with given name (from parameters)
    pm.Deterministic(name_lambda_t, at.exp(lambda_t_log))

    return lambda_t_log


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
                    at.nnet.basic.softplus(tr_len_raw),
                )
                tr_len_list.append(at.nnet.basic.softplus(tr_len_raw))
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
                        at.nnet.basic.softplus(tr_len_L1_raw),
                    )
                    pm.Deterministic(
                        f"transient_len_{i + 1}_hc_L2",
                        at.nnet.basic.softplus(tr_len_L2_raw),
                    )
                else:
                    pm.Deterministic(
                        f"transient_len_{i + 1}", at.nnet.basic.softplus(tr_len_L2_raw)
                    )
                tr_len_list.append(at.nnet.basic.softplus(tr_len_L2_raw))

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
                at.nnet.basic.softplus(tr_len_raw),
            )
            tr_len_list.append(at.nnet.basic.softplus(tr_len_raw))

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
        set_missing_priors_with_default(cp_priors, default_priors_change_points)

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
