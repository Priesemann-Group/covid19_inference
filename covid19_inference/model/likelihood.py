# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-05-19 11:58:38
# @Last Modified: 2020-05-20 10:54:26
# ------------------------------------------------------------------------------ #

import logging

import theano
import theano.tensor as tt
import numpy as np

log = logging.getLogger(__name__)

from .model import *
import pymc3 as pm

def student_t_likelihood(
    new_cases_inferred,
    name_student_t="_new_cases_studentT",
    name_sigma_obs="sigma_obs",
    pr_beta_sigma_obs=30,
    nu=4,
    offset_sigma=1,
    model=None,
    data_obs=None,
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
        ``pr_beta_sigma_obs``. If the input is 2 dimensional, the parameter :math:`\sigma_r` is different for every region.

        Parameters
        ----------
        new_cases_inferred : :class:`~theano.tensor.TensorVariable`
            One or two dimensonal array. If 2 dimensional, the first dimension is time and the second are the
            regions/countries

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
            ``new_cases_inferred`` would lead to very small errors and diverging likelihoods. Defaults to 1.

        model:
            The model on which we want to add the distribution

        data_obs : array
            The data that is observed. By default it is ``model.new_cases_ob``


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

    sigma_obs = pm.HalfCauchy(
        name_sigma_obs, beta=pr_beta_sigma_obs, shape=model.shape_of_regions
    )

    pm.StudentT(
        name=name_student_t,
        nu=nu,
        mu=new_cases_inferred[: len(data_obs)],
        sigma=tt.abs_(new_cases_inferred[: len(data_obs)] + offset_sigma) ** 0.5
        * sigma_obs,  # offset and tt.abs to avoid nans
        observed=data_obs,
    )
