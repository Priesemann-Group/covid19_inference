# ------------------------------------------------------------------------------ #
# Implementation of the Likelihood that is used during the mcmc acceptance
# ------------------------------------------------------------------------------ #

import logging

import pymc as pm
from aesara import scan
import aesara.tensor as at
import numpy as np
from scipy import ndimage as ndi

from .model import *
from . import utility as ut

log = logging.getLogger(__name__)


def student_t_likelihood(
    cases,
    name_student_t="_new_cases_studentT",
    name_sigma_obs="sigma_obs",
    pr_beta_sigma_obs=30,
    nu=4,
    offset_sigma=1,
    model=None,
    data_obs=None,
    sigma_shape=None,
):
    r"""
        Set the likelihood to apply to the model observations (`model.new_cases_obs`)
        We assume a :class:`~pymc3.distributions.continuous.StudentT` distribution because it is robust against outliers [Lange1989]_.
        The likelihood follows:

        .. math::

            P(\text{data\_obs}) &\sim StudentT(\text{mu} = \text{new\_cases\_inferred}, sigma =\sigma,
            \text{nu} = \text{nu})\\
            \sigma &= \sigma_r \sqrt{\text{new\_cases\_inferred} + \text{offset\_sigma}}

        The parameter :math:`\sigma_r` follows
        a :class:`~pymc3.distributions.continuous.HalfCauchy` prior distribution with parameter beta set by
        ``pr_beta_sigma_obs``. If the input is 2 dimensional, the parameter :math:`\sigma_r` is different for every region,
        this can be changed be using the ``sigma_shape`` Parameter.

        Parameters
        ----------
        cases : :class:`~aesara.tensor.TensorVariable`
            The daily new cases estimated by the model.
            Will be compared to  the real world data ``data_obs``.
            One or two dimensonal array. If 2 dimensional, the first dimension is time
            and the second are the regions/countries

        name_student_t :
            The name under which the studentT distribution is saved in the trace.

        name_sigma_obs :
            The name under which the distribution of the observable error is saved in the trace

        pr_beta_sigma_obs : float
            The beta of the :class:`~pymc3.distributions.continuous.HalfCauchy` prior distribution of :math:`\sigma_r`.

        nu : float
            How flat the tail of the distribution is. Larger nu should  make the model
            more robust to outliers. Defaults to 4 [Lange1989]_.

        offset_sigma : float
            An offset added to the sigma, to make the inference procedure robust. Otherwise numbers of
            ``cases`` would lead to very small errors and diverging likelihoods. Defaults to 1.

        model:
            The model on which we want to add the distribution

        data_obs : array
            The data that is observed. By default it is ``model.new_cases_obs``
        
        sigma_shape : int, array
            Shape of the sigma distribution i.e. the data error term. 


        Returns
        -------
        None

        References
        ----------

        .. [Lange1989] Lange, K., Roderick J. A. Little, & Jeremy M. G. Taylor. (1989).
            Robust Statistical Modeling Using the t Distribution.
            Journal of the American Statistical Association,
            84(408), 881-896. doi:10.2307/2290063

    """

    model = modelcontext(model)

    if data_obs is None:
        data_obs = model.new_cases_obs

    if model.shifted_cases:

        no_cases = data_obs == 0
        if len(data_obs.shape) > 1:

            for c in range(data_obs.shape[-1]):
                cases_obs_c = data_obs[..., c]
                # find short intervals of 0 entries and set to NaN
                no_cases_blob, n_blob = ndi.label(no_cases[..., c])
                for i in range(n_blob):
                    if (no_cases_blob == (i + 1)).sum() < 10:
                        data_obs[no_cases_blob == i + 1, ..., c] = np.NaN

                # shift cases from weekends or such to the next day, where cases are reported
                if n_blob > 0:
                    new_cases = 0
                    update = False
                    for i, cases_obs in enumerate(cases_obs_c):
                        new_cases += cases[i + model.diff_data_sim][..., c]
                        if np.isnan(cases_obs):
                            update = True
                        elif update:
                            cases_i = at.set_subtensor(
                                cases[i + model.diff_data_sim][..., c], new_cases
                            )
                            cases = at.set_subtensor(
                                cases[i + model.diff_data_sim], cases_i
                            )
                            new_cases = 0
                            update = False
        else:
            # find short intervals of 0 entries and set to NaN
            no_cases_blob, n_blob = ndi.label(no_cases)
            for i in range(n_blob):
                if (no_cases_blob == (i + 1)).sum() < 10:
                    data_obs[no_cases_blob == i + 1] = np.NaN

            # shift cases from weekends or such to the next day, where cases are reported
            if n_blob > 0:
                new_cases = 0
                update = False
                for i, cases_obs in enumerate(data_obs):
                    new_cases += cases[i + model.diff_data_sim]
                    if np.isnan(cases_obs):
                        update = True
                    elif update:
                        cases = at.set_subtensor(
                            cases[i + model.diff_data_sim], new_cases
                        )
                        new_cases = 0
                        update = False

    cases = cases[model.diff_data_sim : model.data_len + model.diff_data_sim]
    sigma_obs = pm.HalfCauchy(
        name_sigma_obs,
        beta=pr_beta_sigma_obs,
        shape=model.shape_of_regions if sigma_shape is None else sigma_shape,
    )
    sigma = (
        at.abs_(cases + offset_sigma) ** 0.5 * sigma_obs
    )  # offset and at.abs to avoid nans

    pm.StudentT(
        name=name_student_t,
        nu=nu,
        mu=cases[~np.isnan(data_obs)],
        sigma=sigma[~np.isnan(data_obs)],
        observed=data_obs[~np.isnan(data_obs)],
    )
