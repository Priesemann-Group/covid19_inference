# ------------------------------------------------------------------------------ #
# This file provides `delay_cases()` and required helpers:
# Applies delays to time-like arrays (such as a timeseries of observed new cases)
# and adds the required priors and corresponding variables to be traced.
# ------------------------------------------------------------------------------ #

import logging
import numpy as np
import pymc as pm
from aesara import scan
import aesara.tensor as at
from . import utility as ut
from .model import Cov19Model, modelcontext, set_missing_priors_with_default

log = logging.getLogger(__name__)


def delay_cases(
    cases,
    name_delay="delay",
    name_cases=None,
    name_width="delay-width",
    pr_mean_of_median=10,
    pr_sigma_of_median=0.2,
    pr_median_of_width=0.3,
    pr_sigma_of_width=None,
    model=None,
    len_input_arr=None,
    len_output_arr=None,
    diff_input_output=None,
    seperate_on_axes=True,
    num_seperated_axes=None,
    use_gamma=False,
):
    """
    Convolves the input by a lognormal distribution, in order to model a delay:

    * We have a kernel (a distribution) of delays, one realization of this kernel is
      applied to each pymc3 sample.

    * The kernel has a median delay D and a width that correspond to this one
      sample. Doing the ensemble average over all samples and the respective
      kernels, we get two distributions: one of the median delay D and one of the
      width.

    * The (normal) distribution of the median of D is specified using
      `pr_mean_of_median` and `pr_sigma_of_median`.

    * The (lognormal) distribution of the width of the kernel of D is specified
      using `pr_median_of_width` and `pr_sigma_of_width`. If
      `pr_sigma_of_width` is None, the width is fixed (skipping the second
      distribution).

    Parameters
    ----------
    cases : :class:`~aesara.tensor.TensorVariable`
        The input, typically the number of newly infected cases from the output of
        :func:`SIR` or :func:`SEIR`.

    name_delay : str
        The name under which the delay is saved in the trace, suffixes and prefixes
        are added depending on which variable is saved.
        Default : "delay"

    name_cases : str or None
        The name under which the delayed cases are saved in the trace.
        If None, no variable will be added to the trace.
        Default: "delayed_cases"

    pr_mean_of_median : float
        The mean of the :class:`~pymc3.normal` distribution
        which models the prior median of the
        :class:`~pymc3.LogNormal` delay kernel.
        Default: 10.0 (days)

    pr_sigma_of_median : float
        The standart devaiation of :class:`~pymc3.normal`
        distribution which models the prior median of the
        :class:`~pymc3.LogNormal` delay kernel.
        Default: 0.2

    pr_median_of_width : float
        The scale (width) of the :class:`~pymc3.LogNormal`
        delay kernel.
        Default: 0.3

    pr_sigma_of_width : float or None
        Whether to put a prior distribution on the scale (width)
        of the distribution of the delays, too.
        If a number is provided, the scale of the delay kernel follows
        a prior :class:`~pymc3.LogNormal` distribution, with median
        `pr_median_scale_delay` and scale `pr_sigma_scale_delay`.
        Default: None, and no distribution is applied.

    model : :class:`Cov19Model` or None
        The model to use.
        Default: None, model is retrieved automatically from the context


    Other Parameters
    ----------------
    len_input_arr :
        Length of ``new_I_t``. By default equal to ``model.sim_len``. Necessary
        because the shape of aesara tensors are not defined at when the graph is
        built.

    len_output_arr : int
        Length of the array returned. By default it set to the length of the
        cases_obs saved in the model plus the number of days of the forecast.

    diff_input_output : int
        Number of days the returned array begins later then the input. Should be
        significantly larger than the median delay. By default it is set to the
        ``model.diff_data_sim``.

    seperate_on_axes : Bool
        This decides whether or not the delay is applied on every
        axes separately. I.e. Different delay times for the different axes. If
        None **no** axes is modelled separately!

    num_seperated_axes: int or None
        If you are not using separated axes, this is the number of axes.

    Returns
    -------
    delayed_cases : :class:`~aesara.tensor.TensorVariable`
        The delayed input :math:`y_\\text{delayed}(t)`,
        typically the daily number new cases that one expects to measure.
    """
    log.info("Delaying cases")
    model = modelcontext(model)

    # Define the shape of the delays i.e. seperate for different axes
    shape_of_delays = (1,) if not seperate_on_axes else model.shape_of_regions

    # log normal distributed delays (the median values)
    delay_log = pm.Normal(
        name=f"{name_delay}_log",
        mu=np.log(pr_mean_of_median),
        sigma=pr_sigma_of_median,
        shape=shape_of_delays,
    )
    pm.Deterministic(f"{name_delay}", at.exp(delay_log))

    # We may also have a distribution of the width (of the kernel of delays) within
    # each sample/trace.
    if pr_sigma_of_width is None:
        # Default: width of kernel has no distribution, and a shape of (1,)
        width_log = at.as_tensor_variable(np.log(pr_median_of_width))[None]
        width = at.exp(width_log)
    else:
        # Alternatively, put a prior distribution on the witdh, too
        width_log = pm.Normal(
            name=f"{name_width}_log",
            mu=pr_median_of_width,
            sigma=pr_sigma_of_width,
            shape=shape_of_delays,
        )
        # transformation such that it is positive, and not too small:
        width = at.nnet.basic.softplus(width_log) + 0.01
        pm.Deterministic(f"{name_width}", width)

    # enable this function for custom data and data ranges
    if len_output_arr is None:
        len_output_arr = model.sim_len
    if diff_input_output is None:
        diff_input_output = model.diff_data_sim
    if len_input_arr is None:
        len_input_arr = model.sim_len

    # We need to stack the delay_log and width_log
    # depending on the give input shape.
    if cases.ndim >= 2 and not seperate_on_axes:
        delay_log = at.stack([delay_log] * num_seperated_axes, axis=1)
        width = at.stack([width] * num_seperated_axes, axis=1)

    # delay the input cases
    delayed_cases = _delay_lognormal(
        input_arr=cases,
        len_input_arr=len_input_arr,
        len_output_arr=len_output_arr,
        median_delay=at.exp(delay_log),
        scale_delay=width,
        delay_betw_input_output=diff_input_output,
        use_gamma=use_gamma,
    )

    # optionally, add the cases to the trace. maybe let the user do this in the future.
    if name_cases is not None:
        pm.Deterministic(f"{name_cases}", delayed_cases)

    return delayed_cases


def _delay_lognormal(
    input_arr,
    len_input_arr,
    len_output_arr,
    median_delay,
    scale_delay,
    delay_betw_input_output,
    use_gamma=False,
):
    delay_mat = _make_delay_matrix(
        n_rows=len_input_arr,
        n_columns=len_output_arr,
        initial_delay=delay_betw_input_output,
    )
    # avoid negative values that lead to nans in the lognormal distribution
    delay_mat[delay_mat < 0.01] = 0.01

    # add a dim if hierarchical
    if input_arr.ndim == 2:
        delay_mat = delay_mat[:, :, None]
    if input_arr.ndim == 3:
        delay_mat = delay_mat[:, :, None, None]
    delayed_arr = _apply_delay(
        input_arr, median_delay, scale_delay, delay_mat, use_gamma=use_gamma
    )
    return delayed_arr


def _delay_timeshift(new_I_t, len_new_I_t, len_out, delay, delay_diff):
    """
    Delays (time shifts) the input new_I_t by delay.

    Parameters
    ----------
    new_I_t : ~numpy.ndarray or aesara vector
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
    delay_mat = _make_delay_matrix(
        n_rows=len_new_I_t, n_columns=len_out, initial_delay=delay_diff
    )
    inferred_cases = _interpolate(new_I_t, delay, delay_mat)
    return inferred_cases


def _make_delay_matrix(n_rows, n_columns, initial_delay=0):
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


def _apply_delay(array, delay, sigma_delay, delay_mat, use_gamma=False):
    if use_gamma:
        mat = ut.tt_gamma(delay_mat, mu=delay, sigma=sigma_delay)
    else:
        mat = ut.tt_lognormal(delay_mat, mu=np.log(delay), sigma=sigma_delay)
    if array.ndim == 2 and mat.ndim == 3:
        array_shuf = array.dimshuffle((1, 0))
        mat_shuf = mat.dimshuffle((2, 0, 1))
        delayed_arr = at.batched_dot(array_shuf, mat_shuf)
        delayed_arr = delayed_arr.dimshuffle((1, 0))
    elif array.ndim == 1 and mat.ndim == 2:
        delayed_arr = at.dot(array, mat)
    elif array.ndim == 3 and mat.ndim == 4:
        array_flat = array.reshape((array.shape[0], -1))
        mat_flat = mat.reshape((mat.shape[0], mat.shape[1], -1))
        array_shuf = array_flat.dimshuffle((1, 0))
        mat_shuf = mat_flat.dimshuffle((2, 0, 1))
        delayed_arr_flat = at.batched_dot(array_shuf, mat_shuf)
        delayed_arr_flat = delayed_arr_flat.dimshuffle((1, 0))
        delayed_arr = delayed_arr_flat.reshape(array.shape)
    else:
        raise RuntimeError(
            "For some reason, wrong number of dimensions, shouldn't happen"
        )
    return delayed_arr


def _interpolate(array, delay, delay_matrix):
    """
    smooth the array (if delay is no integer)
    """
    if array.ndim == 2:
        interp_matrix = at.maximum(
            1 - at.abs_(at.shape_padaxis(delay_matrix, axis=-1) - delay), 0
        )
        mat_shuf = interp_matrix.dimshuffle((2, 0, 1))
        array_shuf = array.dimshuffle((1, 0))
        delayed_arr = at.batched_dot(array_shuf, mat_shuf)
        interpolation = delayed_arr.dimshuffle((1, 0))
    elif array.ndim == 1:
        interp_matrix = at.maximum(1 - at.abs_(delay_matrix - delay), 0)
        interpolation = at.dot(array, interp_matrix)
    else:
        raise RuntimeError(
            "For some reason, wrong number of dimensions, shouldn't happen"
        )

    return interpolation
