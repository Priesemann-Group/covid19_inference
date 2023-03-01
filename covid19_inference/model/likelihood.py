# ------------------------------------------------------------------------------ #
# Implementation of the Likelihood that is used during the mcmc acceptance
# ------------------------------------------------------------------------------ #

import logging

import pymc as pm
import pytensor.tensor as at
import numpy as np
from scipy import ndimage as ndi

from .model import modelcontext

log = logging.getLogger(__name__)


def student_t_likelihood(
    cases,
    name_student_t="_new_cases_studentT",
    nu=4,
    data_obs=None,
    # Sigma obs
    sigma_obs=None,
    sigma_obs_kwargs={
        "name": "sigma_obs",
        "beta": 30,
    },
    # Other
    offset_sigma=1,
    model=None,
):
    r"""
        Set the likelihood to apply to the model observations (`model.new_cases_obs`)
        We assume a :class:`pymc.StudentT` distribution because it is robust against outliers [Lange1989]_.
        The likelihood follows:

        .. math::

            P(\text{data_obs}) &\sim StudentT(\text{mu} = \text{new_cases_inferred}, sigma =\sigma,
            \text{nu} = \text{nu})\\
            \sigma &= \sigma_r \sqrt{\text{new_cases_inferred} + \text{offset_sigma}}

        The parameter :math:`\sigma_r` follows
        a :class:`pymc.HalfCauchy` prior distribution with parameter beta set by
        ``sigma_obs_kwargs``. If the input is 2 dimensional, the parameter :math:`\sigma_r` is different for every region,
        this can be changed be using the ``sigma_shape`` Parameter.

        Parameters
        ----------

        cases : :class:`~pytensor.tensor.TensorVariable`
            The daily new cases estimated by the model.
            Will be compared to  the real world data ``data_obs``.
            One or two dimensonal array. If 2 dimensional, the first dimension is time
            and the second are the regions/countries

        name_student_t : str, optional
            The name under which the studentT distribution is saved in the trace.

        nu : float, optional
            How flat the tail of the studentT distribution is. Larger nu should  make the model
            more robust to outliers. Defaults to 4 [Lange1989]_.

        data_obs : None or array, optional
            The data that is observed. By default it is ``model.new_cases_obs``
        
        sigma_obs : None or :class:`pymc.Continuous`, optional
            The distribution of the observable error. By default it is a :class:`pymc.HalfCauchy`
            with parameter beta set by ``sigma_obs_kwargs``.

        sigma_obs_kwargs : dict, optional
            The keyword arguments for the observable error distribution if ``sigma_obs`` is None. Defaults to
            ``{"name": "sigma_obs", "beta": 30}``. See :class:`pymc.HalfCauchy` for more options.

        Other Parameters
        ----------------

        offset_sigma : float
            An offset added to the sigma, to make the inference procedure robust. Otherwise numbers of
            ``cases`` would lead to very small errors and diverging likelihoods. Defaults to 1.

        model : None or :class:`Cov19Model`, optional
            The model to use.
            Default: None, model is retrieved automatically from the context

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
    log.info("StudentT likelihood")
    model = modelcontext(model)

    # Check if data_obs is given
    if data_obs is None:
        data_obs = model.new_cases_obs

    # Mask the data
    cases = cases[model.diff_data_sim : model.data_len + model.diff_data_sim]

    # Check if sigma_obs is given
    if sigma_obs is None:
        shape = sigma_obs_kwargs.pop("shape", None)
        sigma_obs = pm.HalfCauchy(
            **sigma_obs_kwargs,
            shape=shape,
        )
        sigma = (
            at.abs(cases + offset_sigma) ** 0.5 * sigma_obs
        )  # offset and at.abs to avoid nans

    # Check if the data is shifted
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

    # StudentT likelihood
    pm.StudentT(
        name=name_student_t,
        nu=nu,
        mu=cases[~np.isnan(data_obs)],
        sigma=sigma[~np.isnan(data_obs)],
        observed=data_obs[~np.isnan(data_obs)],
    )
