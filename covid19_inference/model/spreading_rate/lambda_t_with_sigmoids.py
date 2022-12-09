import logging
import pymc as pm
import numpy as np
import aesara.tensor as at
from .change_points import _make_change_point_RVs
from ..model import modelcontext

log = logging.getLogger(__name__)


def lambda_t_with_sigmoids(
    change_points_list,
    pr_median_lambda_0,
    pr_sigma_lambda_0=0.5,
    sigma_lambda_cp=None,
    sigma_lambda_week_cp=None,
    # Other
    hierarchical=None,
    shape=None,
    name_lambda_t="lambda_t",
    prefix_lambdas="",
    model=None,
):
    r"""
    Builds a time dependent spreading rate :math:`\lambda_t` with change points. The change points are marked by a transient with a sigmoidal shape.

    Parameters
    ----------
    change_points_list : list
        List of change points. Each change point is a dictionary with the following keys
        - "pr_median_lambda": Median of the change point, defaults to value in pr_median_lambda_0
        - "pr_sigma_lambda": Standard deviation of the change point, defaults to value in pr_sigma_lambda_0
        - "pr_mean_date_transient": Transient date (:class:`datetime.datetime`) of the change point
        - "pr_sigma_date_transient": Standard deviation of the transient date (in days), defaults to 2
        - "pr_median_transient_len": Median of the transient length (in days), defaults to 4
        - "pr_sigma_transient_len": Standard deviation of the transient length (in days), defaults to 1
        - "relative_to_previous": If True, the change point is relative to the previous change point, defaults to False
        - "pr_factor_to_previous": Factor to multiply the previous lambda with, defaults to 1

    pr_median_lambda_0 : float
        Median of the initial spreading rate

    pr_sigma_lambda_0 : float, optional
        Standard deviation of the initial spreading rate, defaults to 0.5

    sigma_lambda_cp : None or float, optional
        TODO
    sigma_lambda_week_cp : None or float, optional
        TODO

    Other Parameters
    ----------------
    hierarchical : bool, optional
        Force hierarchical or non-hierarchical model. If None, it is inferred from the model, defaults to None.

    shape : None or tuple, optional
        Shape of the regions. If None, it is inferred from the model, defaults to None.

    name_lambda_t : str, optional
        Name of the lambda_t variable in the trace. If None, no variable is added to the trace, defaults to "lambda_t".

    prefix_lambdas : str, optional
        Prefix for the lambda variables in the trace, defaults to "".

    model : None or :class:`Cov19Model`, optional
        The model to use.
        Default: None, model is retrieved automatically from the context

    Returns
    -------
    lambda_t_log : :class:`aesara.tensor.var.TensorVariable`
        The time dependent spreading rate :math:`\lambda_t` in log space.

    """
    log.info("Lambda_t with sigmoids")
    model = modelcontext(model)

    # If hierarchical is not given, use the value from the model
    if hierarchical is None:
        hierarchical = model.is_hierarchical

    # If shape is not given, use the value from the model
    if shape is None:
        shape = model.shape_of_regions
    if isinstance(shape, int):
        shape = (shape,)

    # ?Get change points random variable?
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
        lambda_t = at.sigmoid((t - tr_time) / tr_len * 4) * (
            lambda_after - lambda_before
        )  # tr_len*4 because the derivative of the sigmoid at zero is 1/4, we want to set it to 1/tr_len

        lambda_before = lambda_after
        lambda_log_t_list.append(lambda_t)

    # Sum up all lambda values from the list
    for i, lambda_t in enumerate(lambda_log_t_list):
        pm.Deterministic(f"{prefix_lambdas}lambda_t_part_{i}", lambda_t)

    lambda_t_log = sum(lambda_log_t_list)

    # Create responding lambda_t pymc variable with given name (from parameters)
    if name_lambda_t is not None:
        pm.Deterministic(name_lambda_t, at.exp(lambda_t_log))

    return lambda_t_log
