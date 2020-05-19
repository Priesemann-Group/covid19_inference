Model
=====

Give an overview how the model is constructed:

* Create an Instance of the base class
* Everything else is attached to the pymc3 model.
* None of our functions actually modifies any data. They rather define ways
  how pymc3 should is allowed to modify data (during the sampling).
* pymc3 context and adding trace variables

Example
-------
.. code-block:: python

    with cov19.model.Cov19Model(**params_model) as model:
        lambda_t_log = cov19.model.lambda_t_with_sigmoids(
            pr_median_lambda_0=0.4,
            pr_sigma_lambda_0=0.5,
            change_points_list=cp_base,
        )

        mu = pm.Lognormal(name="mu", mu=np.log(1 / 8), sigma=0.2)
        pr_median_delay = 10

        new_I_t = cov19.model.SIR(lambda_t_log, mu)

        new_cases_inferred_raw = cov19.model.delay_cases(
            cases=new_I_t,
            pr_mean_of_median=pr_median_delay, pr_median_of_width=0.3
        )

        new_cases_inferred = cov19.model.week_modulation(new_cases_inferred_raw)

        cov19.model.student_t_likelihood(new_cases_inferred)
..


.. contents:: Table of Contents
	:depth: 2

Model Base Class
----------------
.. automodule:: covid19_inference.model.model
    :members:

Compartmental models
--------------------
.. automodule:: covid19_inference.model.compartmental_models
    :members:

Likelihood
----------
.. automodule:: covid19_inference.model.likelihood
    :members:

Delay
-----

@Jonas Let's have the hard math somewhere not directly in the code.

@PS&JD need to update function argument names. Also, when math, then math.

.. math::
    y_\text{delayed}(t) &= \sum_{\tau=0}^T y_\text{ input }(\tau)
    \text{ LogNormal }[
        \log(\text{ delay }), \text{ pr\_median\_scale\_delay }
    ](t - \tau) \\
    \log(\text{ delay }) &= \text{ Normal }(
        \log(\text{ pr\_sigma\_delay } ), \text{ pr\_sigma\_delay }
    )

The `LogNormal` distribution is a function evaluated at :math:`t - \tau`.

If the model is 2-dimensional, the :math:`log(\text{ delay })` is hierarchically
modelled with the :func:`hierarchical_normal` function using the default parameters
except that the prior `sigma` of `delay_L2` is HalfNormal distributed
(``error_cauchy=False``).

.. automodule:: covid19_inference.model.delay
    :members:

Week modulation
---------------
.. automodule:: covid19_inference.model.week_modulation
    :members:


Utility
-------
.. automodule:: covid19_inference.model.utility
    :members:
