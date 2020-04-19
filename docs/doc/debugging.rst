

Debugging
=========


This is some pointer to help debugging models, sampling issues

Sampler: MCMC (Nuts)
--------------------

Divergences
^^^^^^^^^^^

During sampling, a significant fraction of divergences are a sign that the sampler
doesn't sample the whole posterior. In this case the model should be reparametrized.
See this tutorial for a typical example: https://docs.pymc.io/notebooks/Diagnosing_biased_Inference_with_Divergences.html

And these papers include some more details: https://pdfs.semanticscholar.org/7b85/fb48a077c679c325433fbe13b87560e12886.pdf
https://arxiv.org/pdf/1312.0906.pdf

Bad initial energy
^^^^^^^^^^^^^^^^^^

This typically occurs when some distribution in the model can't be evaluated at
the starting point of chain. Run this to see which distribution throws nans or infs:

::

    for RV in model.basic_RVs:
        print(RV.name, RV.logp(model.test_point))

However, this is only evaluates the test_point. When PyMC3 starts sampling, it adds some jitter
around this test_point, which then could lead to nans. Run this to add jitter and then evaluate
the logp:

::

    chains=4
    for RV in model.basic_RVs:
        print(RV.name)

        for _ in range(chains):
            mean = {var: val.copy() for var, val in model.test_point.items()}
            for val in mean.values():
                val[...] += 2 * np.random.rand(*val.shape) - 1
            print(RV.logp(mean))

This code could potentially change in newer versions of PyMC3 (this is tested in 3.8).
Read the source code, to know which random jitter PyMC3 currently adds at beginning.

Nans occur during sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the sampler with the debug mode of Theano.

::

    from theano.compile.nanguardmode import NanGuardMode
    mode = NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False,
                        optimizer='o1')
    trace = pm.sample(mode=mode)


However this doesn't lead to helpful messages if nans occur during gradient evaluations.

Sampler: Variational Inference
------------------------------

There exist some ways to track parameters during sampling. An example:

::

    with model:
        advi = pm.ADVI()
        print(advi.approx.group)

        print(advi.approx.mean.eval())
        print(advi.approx.std.eval())

        tracker = pm.callbacks.Tracker(
            mean=advi.approx.mean.eval,  # callable that returns mean
            std=advi.approx.std.eval  # callable that returns std
        )

        approx = advi.fit(100000, callbacks=[tracker],
                          obj_optimizer=pm.adagrad_window(learning_rate=1e-3),)
                         #total_grad_norm_constraint=10) #constrains maximal gradient, could help


        print(approx.groups[0].bij.rmap(approx.params[0].eval()))

        plt.plot(tracker['mean'])
        plt.plot(tracker['std'])


For the tracker, the order of the parameters is saved in:

::

    approx.ordering.by_name

and the indices encoded there in the slc field.
To plot the mean value of a given parameter name, run:

::

    plt.plot(np.array(tracker['mean'])[:, approx.ordering.by_name['name'].slc]


The debug mode is set with the following parameter:

::

    from theano.compile.nanguardmode import NanGuardMode
    mode = NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False,
                        optimizer='o1')
    approx = advi.fit(100000, callbacks=[tracker],
                 fn_kwargs={'mode':mode})