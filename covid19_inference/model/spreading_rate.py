from .model import *
from .utiility import hierarchical_normal

def _make_change_point_RVs(
    change_points_list, pr_median_lambda_0, pr_sigma_lambda_0=1, model=None
):
    """

    Parameters
    ----------
    priors_dict
    change_points_list
    model

    Returns
    -------

    """

    def hierarchical():
        lambda_0_hc_L2_log, lambda_0_hc_L1_log = hierarchical_normal(
            name = "lambda_0_log_",
            name_sigma = "sigma_lambda_0",
            pr_mean = np.log(pr_median_lambda_0),
            pr_sigma = pr_sigma_lambda_0,
            error_cauchy = False
            )

        pm.Deterministic("lambda_0_hc_L2", tt.exp(ambda_0_hc_L2_log))
        pm.Deterministic("lambda_0_hc_L1", tt.exp(lambda_0_hc_L1_log))
        lambda_log_list.append(lambda_0_hc_L2_log)

        #Create lambda_log list
        for i, cp in enumerate(change_points_list):
            if cp["relative_to_previous"]:
                pr_mean_lambda = lambda_log_list[-1] + tt.log(cp["pr_factor_to_previous"])
            else:
                pr_mean_lambda = np.log(cp["pr_median_lambda"])

            lambda_cp_hc_L2_log, lambda_cp_hc_L1_log = hierarchical_normal(
                name = f"lambda_{i + 1}_log_",
                name_sigma = f"sigma_lambda_{i + 1}",
                pr_mean = pr_mean_lambda,
                pr_sigma = cp["pr_sigma_lambda"],
                error_cauchy = False,
            )
            pm.Deterministic(f"lambda_{i + 1}_hc_L2", tt.exp(lambda_cp_hc_L2_log))
            pm.Deterministic(f"lambda_{i + 1}_hc_L1", tt.exp(lambda_cp_hc_L1_log))
            lambda_log_list.append(lambda_cp_hc_L2_log)

        #Create transient time list
        dt_before = model.sim_begin
        for i, cp in enumerate(change_points_list):
            dt_begin_transient = cp["pr_mean_date_transient"]
            if dt_before is not None and dt_before > dt_begin_transient:
                raise RuntimeError("Dates of change points are not temporally ordered")
            prior_mean = (dt_begin_transient - model.sim_begin).days
            tr_time_L2, _ = hierarchical_normal(
                f"transient_day_{i + 1}",
                f"sigma_transient_day_{i + 1}",
                prior_mean,
                cp["pr_sigma_date_transient"],
                len_L2,
                w=0.5,
                error_cauchy=False,
                error_fact=1.0,
            )
            tr_time_list.append(tr_time_L2)
            dt_before = dt_begin_transient

        #Create transient len list
        for i, cp in enumerate(change_points_list):
            # if model.sim_ndim == 1:
            tr_len_L2_log, tr_len_L1_log = hierarchical_normal(
                f"transient_len_{i + 1}_log",
                f"sigma_transient_len_{i + 1}",
                np.log(cp["pr_median_transient_len"]),
                cp["pr_sigma_transient_len"],
                len_L2,
                w=0.7,
                error_cauchy=False,
            )
            if tr_len_L1_log is not None:
                pm.Deterministic(f"transient_len_{i + 1}_L2", tt.exp(tr_len_L2_log))
                pm.Deterministic(f"transient_len_{i + 1}_L1", tt.exp(tr_len_L1_log))
            else:
                pm.Deterministic(f"transient_len_{i + 1}", tt.exp(tr_len_L2_log))
        tr_len_list.append(tt.exp(tr_len_L2_log))

    def non_hierachical():
        lambda_0_log = pm.Normal(
            name="lambda_0_log_",
            mu=np.log(pr_median_lambda_0),
            sigma=pr_sigma_lambda_0
            )
        pm.Deterministic("lambda_0", tt.exp(lambda_0_log))
        lambda_log_list.append(lambda_0_hc_L2_log)

        #Create lambda_log list
        for i, cp in enumerate(change_points_list):
            if cp["relative_to_previous"]:
                pr_mean_lambda = lambda_log_list[-1] + tt.log(cp["pr_factor_to_previous"])
            else:
                pr_mean_lambda = np.log(cp["pr_median_lambda"])
            lambda_cp_log = pm.Normal(
                name = f"lambda_{i + 1}_log_",
                mu = pr_mean_lambda,
                sigma = cp["pr_sigma_lambda"]
                )
            pm.Deterministic(f"lambda_{i + 1}", tt.exp(lambda_cp_log))
            lambda_log_list.append(lambda_cp_log)

        #Create transient time list
        dt_before = model.sim_begin
        for i, cp in enumerate(change_points_list):        
            dt_begin_transient = cp["pr_mean_date_transient"]
            if dt_before is not None and dt_before > dt_begin_transient:
                raise RuntimeError("Dates of change points are not temporally ordered")

            prior_mean = (dt_begin_transient - model.sim_begin).days
            tr_time = pm.Normal(
                name=f"transient_day_{i + 1}",
                mu=prior_mean,
                sigma=cp["pr_sigma_date_transient"]
                )
            tr_time_list.append(tr_time)
            dt_before = dt_begin_transient

        #Create transient length list
        for i, cp in enumerate(change_points_list):
            tr_len = pm.Normal(
                name = f"transient_len_{i + 1}_log_",
                mu = np.log(cp["pr_median_transient_len"]),
                sigma = cp["pr_sigma_transient_len"]
                )
            pm.Deterministic(f"transient_len_{i + 1}", tt.exp(tr_len_L2_log))

    # ------------------------------------------------------------------------------ #
    # Start of function body
    # ------------------------------------------------------------------------------ #

    default_priors_change_points = dict(
        pr_median_lambda=pr_median_lambda_0,
        pr_sigma_lambda=pr_sigma_lambda_0,
        pr_sigma_date_transient=2,
        pr_median_transient_len=4,
        pr_sigma_transient_len=0.5,
        pr_mean_date_transient=None,
        relative_to_previous=False,
        pr_factor_to_previous=1,
    )

    for cp_priors in change_points_list:
        mh.set_missing_with_default(cp_priors, default_priors_change_points)

    model = modelcontext(model)
    lambda_log_list = []
    tr_time_list = []
    tr_len_list = []

    if model.is_hierarchical:
        hierarchical()
    else:
        non_hierachical()

    return lambda_log_list, tr_time_list, tr_len_list


    """
        TODO
        ----
        def lambda_t_with_transient
    """

def lambda_t_with_sigmoids(
    change_points_list,  pr_median_lambda_0, pr_sigma_lambda_0=0.5, model=None, name_lambda_t="lambda_t"
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
    """

    # Get our default mode context
    model = modelcontext(model)


    lambda_list, tr_time_list, tr_len_list = _make_change_point_RVs(
        change_points_list, pr_median_lambda_0, pr_sigma_lambda_0, model=model
    )

    # model.sim_shape = (time, state)
    # build the time-dependent spreading rate
    lambda_t_list = [lambda_list[0] * tt.ones(model.sim_shape)]
    lambda_before = lambda_list[0]


    for tr_time, tr_len, lambda_after in zip(
        tr_time_list, tr_len_list, lambda_list[1:]
    ):
        t = np.arange(model.sim_shape[0])
        tr_len = tr_len + 1e-5

        if model.is_hierarchical:
            t = np.repeat(t[:, None], model.sim_shape[1], axis=-1)

        lambda_t = tt.nnet.sigmoid((t - tr_time) / tr_len * 4) * (
            lambda_after - lambda_before
        )

        # tr_len*4 because the derivative of the sigmoid at zero is 1/4, we want to set it to 1/tr_len
        lambda_before = lambda_after
        lambda_t_list.append(lambda_t)
    lambda_t_log = sum(lambda_t_list)

    pm.Deterministic(name_lambda_t, tt.exp(lambda_t_log))

    return lambda_t_log


def _smooth_step_function(start_val, end_val, t_begin, t_end, t_total):
    """
        Instead of going from start_val to end_val in one step, make the change a
        gradual linear slope.

        Parameters
        ----------
            start_val : number
                Starting value

            end_val : number
                Target value

            t_begin : number or array (theano)
                Time point (inbetween 0 and t_total) where start_val is placed

            t_end : number or array (theano)
                Time point (inbetween 0 and t_total) where end_val is placed

            t_total : integer
                Total number of time points

        Returns
        -------
            : theano vector
                vector of length t_total with the values of the parameterised f(t)
    """
    t = np.arange(t_total)

    return (
        tt.clip((t - t_begin) / (t_end - t_begin), 0, 1) * (end_val - start_val)
        + start_val
    )
