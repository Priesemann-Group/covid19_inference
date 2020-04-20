import datetime
import platform

import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm
from pymc3 import Model

from . import model_helper as mh


class Cov19_Model(Model):
    """
    Model class used to create a covid-19 propagation dynamics model
    Usage:
    with Cov19_model(params)
    """

    def __init__(self, new_cases_obs, date_begin_data, num_days_forecast,
                 diff_data_sim, N_population, name='', model=None):
        """

        Parameters
        ----------
        new_cases_obs : 1 or 2d array
            If the array is two-dimensional, an hierarchical model will be constructed. First dimension is then time,
            the second the region/country.
        date_begin_data : datatime.datetime
            Date of the first data point
        num_days_forecast : int
            Number of days the simulations runs longer than the data
        diff_data_sim : int
            Number of days the simulation starts earlier than the data. Should be significantly longer than the delay
            between infection and report of cases.
        N_population : number or 1d array
            Number of inhabitance, needed for the S(E)IR model. Is idealy 1 dimensional if new_cases_obs is 2 dimensional
        name : string
            suffix appended to the name of random variables saved in the trace
        model :
            specify a model, if this one should expand another
        """
        super().__init__(name=name, model=model)

        self.new_cases_obs = np.array(new_cases_obs)
        self.ndim_sim = new_cases_obs.ndim
        self.num_days_forecast = num_days_forecast
        len_sim = len(new_cases_obs) + diff_data_sim + num_days_forecast
        if len_sim < len(new_cases_obs) + diff_data_sim:
            raise RuntimeError(
                "Simulation ends before the end of the data. Increase num_days_sim."
            )

        if self.ndim_sim == 1:
            self.shape_sim  = (len_sim,)
        elif self.ndim_sim == 2:
            self.shape_sim = (len_sim, self.new_cases_obs.shape[1])

        date_begin_sim = date_begin_data - datetime.timedelta(days = diff_data_sim)
        self.date_begin_sim = date_begin_sim
        self.diff_data_sim = diff_data_sim
        self.N_population = N_population


def modelcontext(model):
    """
    return the given model or try to find it in the context if there was
    none supplied.
    """
    if model is None:
        return Cov19_Model.get_context()
    return model


def student_t_likelihood(new_cases_inferred, pr_beta_sigma_obs = 30, nu=4, offset_sigma=1, model=None):
    """
    Builds a student-t distribution with mean new_cases_inferred and as observations the new_cases_obs of the model.
    Parameters
    ----------
    new_cases_inferred : 1 or 2d array/random
        If 2 dimensional, the first dimension is time and the second are the regions/countries
    pr_beta_sigma_obs : number

    nu
    offset_sigma
    model

    Returns
    -------

    """
    model = modelcontext(model)

    len_sigma_obs = () if model.ndim_sim == 1 else model.shape_sim[1]
    sigma_obs = pm.HalfCauchy("sigma_obs", beta=pr_beta_sigma_obs, shape=len_sigma_obs)

    num_days_data = model.new_cases_obs.shape[0]
    pm.StudentT(
        name="_new_cases_studentT",
        nu=nu,
        mu=new_cases_inferred[:num_days_data],
        sigma=tt.abs_(new_cases_inferred[:num_days_data]+offset_sigma) ** 0.5 * sigma_obs,  # offset and tt.abs to avoid nans
        observed=model.new_cases_obs,
    )


def SIR(lambda_t_log, pr_beta_I_begin=100, pr_median_mu=1 / 8,
        pr_sigma_mu=0.2, model=None, return_all=False,
        save_all = False):
    """
        Implements the susceptible-infected-recovered model

        Parameters
        ----------
        lambda_t_log : 1 or 2d theano tensor
            time series of the logarithm of the spreading rate

        priors_dict : dict

        model : Cov19_Model:
            if none, retrievs from context

        return_all : Bool
            if True, returns new_I_t, I_t, S_t otherwise returns only new_I_t
        save_all : Bool
            if True, saves new_I_t, I_t, S_t in the trace, otherwise it saves only new_I_t

        Returns
        -------

        new_I_t : array
            time series of the new infected
        I_t : array
            time series of the infected (if return_all set to True)
        S_t : array
            time series of the susceptible (if return_all set to True)

    """
    model = modelcontext(model)

    # Build prior distrubutions:
    mu = pm.Lognormal(
        name="mu",
        mu=np.log(pr_median_mu),
        sigma=pr_sigma_mu,
    )
    N = model.N_population

    num_regions = () if model.ndim_sim == 1 else model.shape_sim[1]

    I_begin = pm.HalfCauchy(name="I_begin", beta=pr_beta_I_begin, shape=num_regions)
    S_begin = N - I_begin

    lambda_t = tt.exp(lambda_t_log)
    new_I_0 = tt.zeros_like(I_begin)

    # Runs SIR model:
    def next_day(lambda_t, S_t, I_t, _, mu, N):
        new_I_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - mu * I_t
        I_t = tt.clip(I_t, 0, N)  # for stability
        S_t = tt.clip(S_t, 0, N)
        return S_t, I_t, new_I_t

    # theano scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, I, new_I
    outputs, _ = theano.scan(
        fn  =next_day,
        sequences=[lambda_t],
        outputs_info=[S_begin, I_begin, new_I_0],
        non_sequences=[mu, N],
    )
    S_t, I_t, new_I_t = outputs
    pm.Deterministic('new_I_t', new_I_t)
    if save_all:
        pm.Deterministic('S_t', S_t)
        pm.Deterministic('I_t', I_t)

    if return_all:
        return new_I_t, I_t, S_t
    else:
        return new_I_t


def delay_cases(new_I_t, pr_median_delay = 10, pr_sigma_delay = 0.2, pr_median_scale_delay = 0.3,
                pr_sigma_scale_delay = None, model=None, save_in_trace=True):
    """

    Parameters
    ----------
    new_I_t
    pr_median_delay
    scale_delay
    pr_sigma_delay
    model

    Returns
    -------

    """


    model = modelcontext(model)

    len_delay = () if model.ndim_sim == 1 else model.shape_sim[1]
    delay_L2_log, delay_L1_log = hierarchical_normal('delay_log', 'sigma_delay',
                                          np.log(pr_median_delay),
                                          pr_sigma_delay,
                                          len_delay,
                                          w=0.9)
    if delay_L1_log is not None:
        pm.Deterministic('delay_L2', np.exp(delay_L2_log))
        pm.Deterministic('delay_L1', np.exp(delay_L1_log))
    else:
        pm.Deterministic('delay', np.exp(delay_L2_log))


    if pr_sigma_scale_delay is not None:
        scale_delay_L2_log, scale_delay_L1_log = hierarchical_normal('scale_delay', 'sigma_scale_delay',
                                                  np.log(pr_median_scale_delay),
                                                  pr_sigma_scale_delay,
                                                  len_delay,
                                                  w=0.9)
        pm.Deterministic('scale_delay_L2', np.exp(scale_delay_L2_log))
        pm.Deterministic('scale_delay_L1', np.exp(scale_delay_L1_log))

    else:
        scale_delay_L2_log=np.log(pr_median_scale_delay)
        pm.Deterministic('scale_delay', np.exp(scale_delay_L2_log))

    num_days_sim = model.shape_sim[0]
    diff_data_sim = model.diff_data_sim
    new_cases_inferred = mh.delay_cases_lognormal(input_arr=new_I_t,
                                                len_input_arr=num_days_sim,
                                                len_output_arr=num_days_sim - diff_data_sim,
                                                median_delay=tt.exp(delay_L2_log),
                                                scale_delay=tt.exp(scale_delay_L2_log),
                                                delay_betw_input_output=diff_data_sim)
    if save_in_trace:
        pm.Deterministic('new_cases_raw', new_cases_inferred)

    return new_cases_inferred


def week_modulation(new_cases_inferred, week_modulation_type='abs_sine', pr_mean_weekend_factor=0.7,
                    pr_sigma_weekend_factor=0.2, week_end_days = (6,7), model=None, save_in_trace=True):
    """

    Parameters
    ----------
    new_cases_inferred
    week_modulation_type
    pr_mean_weekend_factor
    pr_sigma_weekend_factor
    week_end_days
    model

    Returns
    -------

    """
    model = modelcontext(model)
    shape_modulation = list(model.shape_sim)
    diff_data_sim = model.diff_data_sim
    shape_modulation[0] -= diff_data_sim
    date_begin_sim = model.date_begin_sim

    len_L2 = () if model.ndim_sim == 1 else model.shape_sim[1]


    week_end_factor, _ = hierarchical_normal('weekend_factor', 'sigma_weekend_factor',
                                        pr_mean=pr_mean_weekend_factor,
                                        pr_sigma=pr_sigma_weekend_factor,
                                        len_L2=len_L2)
    if week_modulation_type == 'step':
        modulation = np.zeros(shape_modulation[0])
        for i in range(shape_modulation[0]):
            date_curr = date_begin_sim + datetime.timedelta(days=i + diff_data_sim + 1)
            if date_curr.isoweekday() in week_end_days:
                modulation[i] = 1
    elif week_modulation_type == 'abs_sine':
        offset_rad = pm.VonMises('offset_modulation_rad', mu=0, kappa=0.01)
        offset = pm.Deterministic('offset_modulation', offset_rad / (2 * np.pi) * 7)
        t = np.arange(shape_modulation[0])
        date_begin = date_begin_sim + datetime.timedelta(days=diff_data_sim + 1)
        weekday_begin = date_begin.weekday()
        t -= weekday_begin  # Sunday is zero
        modulation = 1 - tt.abs_(tt.sin(t / 7 * np.pi + offset_rad / 2))

    if model.ndim_sim == 2:
        modulation = tt.shape_padaxis(modulation, axis=-1)

    multiplication_vec = np.ones(shape_modulation) - (1 - week_end_factor) * modulation
    new_cases_inferred_eff = new_cases_inferred * multiplication_vec
    if save_in_trace:
        pm.Deterministic('new_cases', new_cases_inferred_eff)
    return new_cases_inferred_eff


def make_change_point_RVs(change_points_list, pr_median_lambda_0, pr_sigma_lambda_0=1, model=None):
    """

    Parameters
    ----------
    priors_dict
    change_points_list
    model

    Returns
    -------

    """

    default_priors_change_points = dict(
        pr_median_lambda=pr_median_lambda_0,
        pr_sigma_lambda=pr_sigma_lambda_0,
        pr_sigma_date_transient=3,
        pr_median_transient_len=4,
        pr_sigma_transient_len=1,
        pr_mean_date_transient=None,
    )


    for cp_priors in change_points_list:
        mh.set_missing_with_default(cp_priors, default_priors_change_points)

    model = modelcontext(model)
    len_L2 = () if model.ndim_sim == 1 else model.shape_sim[1]


    lambda_log_list = []
    tr_time_list = []
    tr_len_list = []

    #
    lambda_0_L2_log, lambda_0_L1_log = hierarchical_normal('lambda_0_log', 'sigma_lambda_0',
                                             np.log(pr_median_lambda_0),
                                             pr_sigma_lambda_0,
                                             len_L2, w=0.4,
                                                error_cauchy=False)
    if lambda_0_L1_log is not None:
        pm.Deterministic('lambda_0_L2', np.exp(lambda_0_L2_log))
        pm.Deterministic('lambda_0_L1', np.exp(lambda_0_L1_log))
    else:
        pm.Deterministic('lambda_0', np.exp(lambda_0_L2_log))

    lambda_log_list.append(lambda_0_L2_log)
    for i, cp in enumerate(change_points_list):
        lambda_cp_L2_log, lambda_cp_L1_log = hierarchical_normal(f'lambda_{i + 1}_log',
                                               f'sigma_lambda_{i + 1}',
                                               np.log(cp["pr_median_lambda"]),
                                               cp["pr_sigma_lambda"],
                                               len_L2,
                                               w=0.7,
                                                error_cauchy=False)
        if lambda_cp_L1_log is not None:
            pm.Deterministic(f'lambda_{i + 1}_L2', np.exp(lambda_cp_L2_log))
            pm.Deterministic(f'lambda_{i + 1}_L1', np.exp(lambda_cp_L1_log))
        else:
            pm.Deterministic(f'lambda_{i + 1}', np.exp(lambda_cp_L2_log))

        lambda_log_list.append(lambda_cp_L2_log)


    dt_before = model.date_begin_sim
    for i, cp in enumerate(change_points_list):
        dt_begin_transient = cp["pr_mean_date_transient"]
        if dt_before is not None and dt_before > dt_begin_transient:
            raise RuntimeError("Dates of change points are not temporally ordered")
        prior_mean = (
                dt_begin_transient - model.date_begin_sim
        ).days
        tr_time_L2, = hierarchical_normal(f'transient_day_{i + 1}',
                                               f'sigma_transient_day_{i + 1}',
                                               prior_mean,
                                               cp["pr_sigma_date_transient"],
                                               len_L2,
                                               w=0.5)


        tr_time_list.append(tr_time_L2)


    for i, cp in enumerate(change_points_list):
        #if model.ndim_sim == 1:
        tr_len_L2_log, tr_len_L1_log = hierarchical_normal(f'transient_len_{i + 1}_log',
                                             f'sigma_transient_len_{i + 1}',
                                             np.log(cp["pr_median_transient_len"]),
                                             cp["pr_sigma_transient_len"],
                                             len_L2,
                                             w=0.7)
        if tr_len_L1_log is not None:
            pm.Deterministic(f'transient_len_{i + 1}_L2', np.exp(tr_len_L2_log))
            pm.Deterministic(f'transient_len_{i + 1}_L1', np.exp(tr_len_L1_log))
        else:
            pm.Deterministic(f'transient_len_{i + 1}', np.exp(tr_len_L2_log))

        tr_len_list.append(tt.exp(tr_len_L2_log))
    return lambda_log_list, tr_time_list, tr_len_list

def lambda_t_with_sigmoids(change_points_list, pr_median_lambda_0, pr_sigma_lambda_0=1, model=None):
    """

    Parameters
    ----------
    change_points_list
    pr_median_lambda_0
    pr_sigma_lambda_0
    model

    Returns
    -------

    """


    model = modelcontext(model)
    shape_sim = model.shape_sim

    lambda_list, tr_time_list, tr_len_list = make_change_point_RVs(change_points_list, pr_median_lambda_0, pr_sigma_lambda_0, model=model)

    num_days_sim = shape_sim[0]

    # build the time-dependent spreading rate
    if len(shape_sim) == 2:
        lambda_t_list = [lambda_list[0] * tt.ones(shape_sim)]
    else:
        lambda_t_list = [lambda_list[0] * tt.ones(shape_sim)]
    lambda_before = lambda_list[0]

    for tr_time, tr_len, lambda_after in zip(
            tr_time_list, tr_len_list, lambda_list[1:]
    ):
        t = np.arange(num_days_sim)
        tr_len = tr_len + 1e-5
        if len(shape_sim) == 2:
            t = np.repeat(t[:, None], shape_sim[1], axis=-1)
        lambda_t = tt.nnet.sigmoid((t - tr_time) / tr_len * 4) * (lambda_after - lambda_before)
        # tr_len*4 because the derivative of the sigmoid at zero is 1/4, we want to set it to 1/tr_len
        lambda_before = lambda_after
        lambda_t_list.append(lambda_t)
    lambda_t_log = sum(lambda_t_list)

    pm.Deterministic('lambda_t', tt.exp(lambda_t_log))
    
    return lambda_t_log


def hierarchical_normal(name, name_sigma, pr_mean, pr_sigma, len_L2, w=1.0, error_fact = 2.,
                        error_cauchy=True):
    """
    Takes ideas from https://arxiv.org/pdf/1312.0906.pdf (see also https://arxiv.org/pdf/0708.3797.pdf and
     https://pdfs.semanticscholar.org/7b85/fb48a077c679c325433fbe13b87560e12886.pdf)
    and https://projecteuclid.org/euclid.ba/1340371048 chapter 6
    Parameters
    ----------
    name_Y
    name_X
    name_sigma_Y
    pr_mean
    pr_sigma
    pr_beta
    len_Y
    w

    Returns
    -------

    """
    if not len_L2: # not hierarchical
        Y = pm.Normal(name, mu=pr_mean, sigma=pr_sigma)
        return Y, None
    else:
        w=1.
        if error_cauchy:
            sigma_Y = pm.HalfCauchy(name_sigma + '_L2', beta=error_fact * pr_sigma)
        else:
            sigma_Y = pm.HalfNormal(name_sigma + '_L2', sigma=error_fact * pr_sigma)


        X = pm.Normal(name + '_L1' , mu=pr_mean, sigma=pr_sigma)
        phi = pm.Normal(name + '_L2_raw', mu=0, sigma=1, shape=len_L2)  # (1-w**2)*sigma_X+1*w**2, shape=len_Y)
        Y = w * X + phi*sigma_Y
        pm.Deterministic(name + '_L2', Y)
        return Y, X


def hierarchical_beta(name, name_sigma, pr_mean, pr_sigma, len_L2):

    if not len_L2: # not hierarchical
        Y = pm.Beta(name, alpha=pr_mean/pr_sigma, beta=1/pr_sigma*(1-pr_mean))
        return Y, None
    else:
        sigma_Y = pm.HalfCauchy(name_sigma + '_L2', beta=pr_sigma)
        X = pm.Beta(name + '_L1' , alpha=pr_mean/pr_sigma, beta=1/pr_sigma*(1-pr_mean))
        Y = pm.Beta(name + '_L2', alpha=X/sigma_Y, beta=1/sigma_Y*(1-X), shape=len_L2)
        return Y, X