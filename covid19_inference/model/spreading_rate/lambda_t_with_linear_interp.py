import logging
import pymc as pm
import aesara.tensor as at
from .change_points import _make_change_point_RVs
from ..model import modelcontext

log = logging.getLogger(__name__)


def lambda_t_with_linear_interp(
    change_points_list,
    pr_median_lambda_0,
    pr_sigma_lambda_0=0.5,
    # Other
    name_lambda_t="lambda_t",
    model=None,
):
    r"""
    Builds a time dependent spreading rate :math:`\lambda_t` with change points. The change points are marked by a transient with a linear interpolation.

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

    Returns
    -------
    lambda_t_log : :class:`aesara.tensor.var.TensorVariable`
        The time dependent spreading rate :math:`\lambda_t` in log space.

    Other Parameters
    ----------------

    name_lambda_t : str, optional
        Name of the lambda_t variable in the trace. If None, no variable is added to the trace, defaults to "lambda_t".

    model : :class:`Cov19Model`, optional
        Is retrieved from the context by default.

    """
    log.info("Lambda_t linear in lin-space")
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

    # Create responding lambda_t pymc variable with given name (from parameters)
    if name_lambda_t is not None:
        pm.Deterministic(name_lambda_t, at.exp(lambda_t_log))

    return lambda_t_log
