# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-05-19 11:57:38
# @Last Modified: 2020-05-19 12:02:48
# ------------------------------------------------------------------------------ #

import logging

import theano
import theano.tensor as tt
import numpy as np

log = logging.getLogger(__name__)

# delay.py
# fix names, hc_fix
def delay_cases(
    new_I_t,
    name_delay="delay",
    name_delayed_cases="new_cases_raw",
    pr_median_delay=10,
    pr_sigma_median_delay=0.2,
    pr_median_scale_delay=0.3,
    pr_sigma_scale_delay=None,
    model=None,
    save_in_trace=True,
    len_input_arr=None,
    len_output_arr=None,
    diff_input_output=None,
):
    r"""
        Convolves the input by a lognormal distribution, in order to model a delay:

        .. math::

            y_\text{delayed}(t) &= \sum_{\tau=0}^T y_\text{input}(\tau) LogNormal[log(\text{delay}), \text{pr\_median\_scale\_delay}](t - \tau)\\
            log(\text{delay}) &= Normal(log(\text{pr\_sigma\_delay}), \text{pr\_sigma\_delay})

        For clarification: the :math:`LogNormal` distribution is a function evaluated at :math:`t - \tau`.

        If the model is 2-dimensional, the :math:`log(\text{delay})` is hierarchically modelled with the
        :func:`hierarchical_normal` function using the default parameters except that the
        prior :math:`\sigma` of :math:`\text{delay}_\text{L2}` is HalfNormal distributed (``error_cauchy=False``).


        Parameters
        ----------
        new_I_t : :class:`~theano.tensor.TensorVariable`
            The input, typically the number newly infected cases :math:`I_{new}(t)` of from the output of
            :func:`SIR` or :func:`SEIR`.
        name_delay : str
            The name under which the delay is saved in the trace, suffixes and prefixes are added depending on which
            variable is saved.
        name_delayed_cases : str
            The name under which the delay is saved in the trace, suffixes and prefixes are added depending on which
            variable is saved.
        pr_median_delay : float
            The mean of the :class:`~pymc3.distributions.continuous.normal` distribution which
            models the prior median of the :class:`~pymc3.distributions.continuous.LogNormal` delay kernel.
        pr_sigma_median_delay : float
            The standart devaiation of :class:`~pymc3.distributions.continuous.normal` distribution which
            models the prior median of the :class:`~pymc3.distributions.continuous.LogNormal` delay kernel.
        pr_median_scale_delay : float
            The scale (width) of the :class:`~pymc3.distributions.continuous.LogNormal` delay kernel.
        pr_sigma_scale_delay : float
            If it is not None, the scale is of the delay is kernel follows a prior
            :class:`~pymc3.distributions.continuous.LogNormal` distribution, with median ``pr_median_scale_delay`` and
            scale ``pr_sigma_scale_delay``.
        model : :class:`Cov19Model`
            if none, it is retrieved from the context
        save_in_trace : bool
            whether to save :math:`y_\text{delayed}` in the trace
        len_input_arr :
            Length of ``new_I_t``. By default equal to ``model.sim_len``. Necessary because the shape of theano
            tensors are not defined at when the graph is built.
        len_output_arr : int
            Length of the array returned. By default it set to the length of the cases_obs saved in the model plus
            the number of days of the forecast.
        diff_input_output : int
            Number of days the returned array begins later then the input. Should be significantly larger than
            the median delay. By default it is set to the ``model.diff_data_sim``.

        Returns
        -------
        new_cases_inferred : :class:`~theano.tensor.TensorVariable`
            The delayed input :math:`y_\text{delayed}(t)`, typically the daily number new cases that one expects to measure.
    """

    model = modelcontext(model)

    if len_output_arr is None:
        len_output_arr = model.data_len + model.fcast_len
    if diff_input_output is None:
        diff_input_output = model.diff_data_sim
    if len_input_arr is None:
        len_input_arr = model.sim_len

    len_delay = () if model.sim_ndim == 1 else model.sim_shape[1]
    delay_L2_log, delay_L1_log = hierarchical_normal(
        name_delay + "_log",
        "sigma_" + name_delay,
        np.log(pr_median_delay),
        pr_sigma_median_delay,
        len_delay,
        w=0.9,
        error_cauchy=False,
    )
    if delay_L1_log is not None:
        pm.Deterministic(f"{name_delay}_L2", np.exp(delay_L2_log))
        pm.Deterministic(f"{name_delay}_L1", np.exp(delay_L1_log))
    else:
        pm.Deterministic(f"{name_delay}", np.exp(delay_L2_log))

    if pr_sigma_scale_delay is not None:
        scale_delay_L2_log, scale_delay_L1_log = hierarchical_normal(
            "scale_" + name_delay,
            "sigma_scale_" + name_delay,
            np.log(pr_median_scale_delay),
            pr_sigma_scale_delay,
            len_delay,
            w=0.9,
            error_cauchy=False,
        )
        if scale_delay_L1_log is not None:
            pm.Deterministic(f"scale_{name_delay}_L2", tt.exp(scale_delay_L2_log))
            pm.Deterministic(f"scale_{name_delay}_L1", tt.exp(scale_delay_L1_log))

        else:
            pm.Deterministic(f"scale_{name_delay}", tt.exp(scale_delay_L2_log))
    else:
        scale_delay_L2_log = np.log(pr_median_scale_delay)

    new_cases_inferred = mh.delay_cases_lognormal(
        input_arr=new_I_t,
        len_input_arr=len_input_arr,
        len_output_arr=len_output_arr,
        median_delay=tt.exp(delay_L2_log),
        scale_delay=tt.exp(scale_delay_L2_log),
        delay_betw_input_output=diff_input_output,
    )
    if save_in_trace:
        pm.Deterministic(name_delayed_cases, new_cases_inferred)

    return new_cases_inferred


# underscore
# delay.py
def delay_timeshift(new_I_t, len_new_I_t, len_out, delay, delay_diff):
    """
        Delays (time shifts) the input new_I_t by delay.

        Parameters
        ----------
        new_I_t : ~numpy.ndarray or theano vector
            Input to be delayed.

        len_new_I_t : integer
            Length of new_I_t. (Think len(new_I_t) ).
            Assure len_new_I_t is larger then len(new_cases_obs)-delay, otherwise it
            means that the simulated data is not long enough to be fitted to the data.

        len_out : integer
            Length of the output.

        delay : number
            If delay is an integer, the array will be exactly shifted. Else, the data
            will be shifted and intepolated (convolved with hat function of width one).
            Take care that delay is smaller than or equal to delay_diff,
            otherwise zeros are returned, which could potentially lead to errors.

        delay_diff: integer
            The difference in length between the new_I_t and the output.

        Returns
        -------
            an array with length len_out that was time-shifted by delay
    """

    # elementwise delay of input to output
    delay_mat = make_delay_matrix(
        n_rows=len_new_I_t, n_columns=len_out, initial_delay=delay_diff
    )
    inferred_cases = interpolate(new_I_t, delay, delay_mat)
    return inferred_cases


# underscore
# delay.py
def make_delay_matrix(n_rows, n_columns, initial_delay=0):
    """
        Has in each entry the delay between the input with size n_rows and the output
        with size n_columns

        initial_delay is the top-left element.
    """
    size = max(n_rows, n_columns)
    mat = np.zeros((size, size))
    for i in range(size):
        diagonal = np.ones(size - i) * (initial_delay + i)
        mat += np.diag(diagonal, i)
    for i in range(1, size):
        diagonal = np.ones(size - i) * (initial_delay - i)
        mat += np.diag(diagonal, -i)
    return mat[:n_rows, :n_columns]


# underscore
# delay.py
def apply_delay(array, delay, sigma_delay, delay_mat):
    mat = tt_lognormal(delay_mat, mu=np.log(delay), sigma=sigma_delay)
    if array.ndim == 2 and mat.ndim == 3:
        array_shuf = array.dimshuffle((1, 0))
        mat_shuf = mat.dimshuffle((2, 0, 1))
        delayed_arr = tt.batched_dot(array_shuf, mat_shuf)
        delayed_arr = delayed_arr.dimshuffle((1, 0))
    elif array.ndim == 1 and mat.ndim == 2:
        delayed_arr = tt.dot(array, mat)
    else:
        raise RuntimeError(
            "For some reason, wrong number of dimensions, shouldn't happen"
        )
    return delayed_arr


# underscore
# delay.py
def delay_lognormal(
    input_arr,
    len_input_arr,
    len_output_arr,
    median_delay,
    scale_delay,
    delay_betw_input_output,
):
    delay_mat = make_delay_matrix(
        n_rows=len_input_arr,
        n_columns=len_output_arr,
        initial_delay=delay_betw_input_output,
    )
    delay_mat[
        delay_mat < 0.01
    ] = 0.01  # needed because negative values lead to nans in the lognormal distribution.
    if input_arr.ndim == 2:
        delay_mat = delay_mat[:, :, None]
    delayed_arr = apply_delay(input_arr, median_delay, scale_delay, delay_mat)
    return delayed_arr


# delay.py
# underscore this
def interpolate(array, delay, delay_matrix):
    """
        smooth the array (if delay is no integer)
    """
    interp_matrix = tt.maximum(1 - tt.abs_(delay_matrix - delay), 0)
    interpolation = tt.dot(array, interp_matrix)
    return interpolation
