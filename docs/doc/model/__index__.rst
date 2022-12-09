Model
=====

Overview
--------

If you are familiar with ``pymc``, then looking at the example
below should explain how to construct a model. Otherwise, here is a quick overview:

1. First we create a model object, which is a container for the
model variables and their relationships. This object has 
some convenience properties to get the range of the data, 
simulation length and so on.

2. We than use that model object and add details to it. These
details correspond to the actual (physical) model features, 
such as the spreading dynamics, the initial conditions, and
week modulations. We supply a number of pre-defined model
features which you can combine to create your own model.

    * Every feature has its own function that takes in arguments to set prior assumptions.
    * Most of these functions add variables to the ``trace`` which can be used after the sampling process to analyze the results.

3. After construction of the model, we can sample from the
posterior distribution of the model parameters. This is done using
pymc sampling functions.

Have a look at the following :ref:`example<Example>` to see how this might look as code.


Model features can be added using your own functions or using the pre-defined functions in the ``model`` module.

.. toctree::
    :caption: Model module
    :maxdepth: 2

    spreading_rate
    compartmental_models
    delay
    week_modulation
    likelihood
    

Example
-------

Here is a simple example of how to use the model module to create a simple model and sample from it. If you are looking for a more
in-depth example, have a look at the `example notebooks <https://github.com/Priesemann-Group/covid19_inference/tree/master/scripts/interactive>`_.


.. code-block:: python

    from datetime import datetime
    import pymc as pm
    import covid19_inference as cov19

    # Load the data
    begin_date = datetime(2020, 3, 1)
    end_date = datetime(2020, 4, 1)

    # Download data 
    jhu = cov19.data_retrieval.JHU()
    new_cases = jhu.get_new(
        country="Germany",
        begin_date=begin_date,
        end_date=end_date
    )

    # change points (see Dehning 2020)  
    change_points = [
        dict(pr_mean_date_transient=datetime(2020, 3, 9)),
        dict(pr_mean_date_transient=datetime(2020, 3, 16)),
        dict(pr_mean_date_transient=datetime(2020, 3, 23)),
    ]

    # Create the model
    with cov19.model.Cov19Model(new_cases) as model:

        # apply change points, lambda is in log scale
        lambda_t_log = cov19.model.lambda_t_with_sigmoids(
            pr_median_lambda_0=0.4,
            pr_sigma_lambda_0=0.5,
            change_points_list=change_points,
        )

        # set prior distribution for the recovery rate
        mu = pm.LogNormal(name="mu", mu=np.log(1 / 8), sigma=0.2)

        # Use lambda_t_log and mu to run a SIR model
        new_I_t = cov19.model.SIR(lambda_t_log, mu)

        # Delay the cases to account for reporting delay
        new_cases_inferred_raw = cov19.model.delay_cases(
            cases=new_I_t,
            pr_mean_of_median=10,
        )

        # apply a weekly modulation i.e. fewer reports during weekends
        new_cases_inferred = cov19.model.week_modulation(new_cases_inferred_raw)

        # set the likeliehood
        cov19.model.student_t_likelihood(new_cases_inferred)


    # Sample from the posterior distribution
    trace = pm.sample(model=model, tune=1000, draws=1000, init="advi+adapt_diag")


Model Base Class
----------------

.. autoclass:: covid19_inference.model.Cov19Model
    :members:
