Model
=====

If you are familiar with ``pymc3``, then looking at the example below should explain
how our model works. Otherwise, here is a quick overivew:

* First, we have to create an instance of the base class (that is derived from pymc3s model class). It has some convenient properties to get the range of the data, simulation length and so forth.
* We then add details that base model. They correspond to the actual (physical) model features, such as the change points, the reporting delay and the week modulation.

    - Every feature has it's own function that takes in arguments to set prior
      assumptions.
    - Sometimes they also take in input (data, reported cases ... ) but none of the
      function performs any actual modifactions on the data. They only tell pymc3 what
      it is supposed to do during the sampling.
    - None of our functions actually modifies any data. They rather define ways how
      pymc3 should modify data during the sampling.
    - Most of the feature functions add variables to the ``pymc3.trace``, see the function arguments that start with ``name_``.

* in pymc3 it is common to use a context, as we also do in the example. everything within the block ``with cov19.model.Cov19Model(**params_model) as this_model:`` automagically applies to ``this_model``. Alternatively, you could provide a keyword to each function ``model=this_model``.

Example
-------
.. code-block:: python

    import datetime

    import pymc3 as pm
    import numpy as np
    import covid19_inference as cov19

    # limit the data range
    bd = datetime.datetime(2020, 3, 2)

    # download data
    jhu = cov19.data_retrieval.JHU(auto_download=True)
    new_cases = jhu.get_new(country="Germany", data_begin=bd)

    # set model parameters
    params_model = dict(
        new_cases_obs=new_cases,
        data_begin=bd,
        fcast_len=28,
        diff_data_sim=16,
        N_population=83e6,
    )

    # change points like in the paper
    change_points = [
        dict(pr_mean_date_transient=datetime.datetime(2020, 3, 9)),
        dict(pr_mean_date_transient=datetime.datetime(2020, 3, 16)),
        dict(pr_mean_date_transient=datetime.datetime(2020, 3, 23)),
    ]

    # create model instance and add details
    with cov19.model.Cov19Model(**params_model) as this_model:
        # apply change points, lambda is in log scale
        lambda_t_log = cov19.model.lambda_t_with_sigmoids(
            pr_median_lambda_0=0.4,
            pr_sigma_lambda_0=0.5,
            change_points_list=change_points,
        )

        # prior for the recovery rate
        mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)

        # new Infected day over day are determined from the SIR model
        new_I_t = cov19.model.SIR(lambda_t_log, mu)

        # model the reporting delay, our prior is ten days
        new_cases_inferred_raw = cov19.model.delay_cases(
            cases=new_I_t,
            pr_mean_of_median=10,
        )

        # apply a weekly modulation, fewer reports during weekends
        new_cases_inferred = cov19.model.week_modulation(new_cases_inferred_raw)

        # set the likeliehood
        cov19.model.student_t_likelihood(new_cases_inferred)

    # run the sampling
    trace = pm.sample(model=this_model, tune=50, draws=10, init="advi+adapt_diag")
..


.. contents:: Table of Contents
	:depth: 2

Model Base Class
----------------
.. autoclass:: covid19_inference.model.Cov19Model
    :members:

Compartmental models
--------------------

SIR --- susceptible-infected-recovered
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: covid19_inference.model.SIR

More Details
^^^^^^^^^^^^

.. math::
    I_{new}(t) &= \lambda_t I(t-1)  \frac{S(t-1)}{N}   \\
    S(t) &= S(t-1) - I_{new}(t)  \\
    I(t) &= I(t-1) + I_{new}(t) - \mu  I(t)

The prior distributions of the recovery rate :math:`\mu`
and :math:`I(0)` are set to

.. math::
    \mu &\sim \text{LogNormal}\left[
            \log(\text{pr\_median\_mu}), \text{pr\_sigma\_mu}
        \right] \\
    I(0) &\sim \text{HalfCauchy}\left[
            \text{pr\_beta\_I\_begin}
        \right]

---------------------------------------

SEIR-like ---  susceptible-exposed-infected-recovered
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: covid19_inference.model.SEIR

More Details
^^^^^^^^^^^^

.. math::
    E_{\text{new}}(t) &= \lambda_t I(t-1) \frac{S(t)}{N}   \\
    S(t) &= S(t-1) - E_{\text{new}}(t)  \\
    I_\text{new}(t) &= \sum_{k=1}^{10} \beta(k) E_{\text{new}}(t-k)   \\
    I(t) &= I(t-1) + I_{\text{new}}(t) - \mu  I(t) \\
    \beta(k) & = P(k) \sim \text{LogNormal}\left[
            \log(d_{\text{incubation}}), \text{sigma\_incubation}
        \right]

The recovery rate :math:`\mu` and the incubation period is the same for all regions and follow respectively:

.. math::
     P(\mu) &\sim \text{LogNormal}\left[
            \text{log(pr\_median\_mu), pr\_sigma\_mu}
        \right] \\
     P(d_{\text{incubation}}) &\sim \text{Normal}\left[
            \text{pr\_mean\_median\_incubation, pr\_sigma\_median\_incubation}
        \right]

The initial number of infected and newly exposed differ for each region and follow prior :class:`~pymc3.distributions.continuous.HalfCauchy` distributions:

.. math::
    E(t)  &\sim \text{HalfCauchy}\left[
            \text{pr\_beta\_E\_begin}
        \right] \:\: \text{for} \: t \in {-9, -8, ..., 0}\\
    I(0)  &\sim \text{HalfCauchy}\left[
            \text{pr\_beta\_I\_begin}
        \right].

References
^^^^^^^^^^

* .. [Nishiura2020]
    Nishiura, H.; Linton, N. M.; Akhmetzhanov, A. R.
    Serial Interval of Novel Coronavirus (COVID-19) Infections.
    Int. J. Infect. Dis. 2020, 93, 284–286. https://doi.org/10.1016/j.ijid.2020.02.060.
* .. [Lauer2020]
    Lauer, S. A.; Grantz, K. H.; Bi, Q.; Jones, F. K.; Zheng, Q.; Meredith, H. R.; Azman, A. S.; Reich, N. G.; Lessler, J.
    The Incubation Period of Coronavirus Disease 2019 (COVID-19) From Publicly Reported Confirmed Cases: Estimation and Application.
    Ann Intern Med 2020. https://doi.org/10.7326/M20-0504.

---------------------------------------
.. autofunction:: covid19_inference.model.uncorrelated_prior_I


Likelihood
----------
.. autofunction:: covid19_inference.model.student_t_likelihood


Spreading Rate
--------------
.. autofunction:: covid19_inference.model.lambda_t_with_sigmoids

Delay
-----
.. autofunction:: covid19_inference.model.delay_cases

More Details
^^^^^^^^^^^^

.. math::
    y_\text{delayed}(t) &= \sum_{\tau=0}^T y_\text{input}(\tau)
    \text{LogNormal}\left[
        \log(\text{delay}), \text{pr\_median\_scale\_delay}
    \right](t - \tau) \\
    \log(\text{delay}) &= \text{Normal}\left[
        \log(\text{pr\_sigma\_delay} ), \text{pr\_sigma\_delay}
    \right]

The `LogNormal` distribution is a function evaluated at :math:`t - \tau`.

If the model is 2-dimensional (hierarchical), the :math:`\log(\text{delay})` is hierarchically
modelled with the :func:`hierarchical_normal` function using the default parameters
except that the prior `sigma` of `delay_L2` is HalfNormal distributed
(``error_cauchy=False``).



Week modulation
---------------
.. autofunction:: covid19_inference.model.week_modulation



Utility
-------
.. automodule:: covid19_inference.model.utility
    :members:
