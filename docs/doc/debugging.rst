

Debugging
=========


This is some pointer to help debugging models and sampling issues


General approach for nans/infs during sampling
----------------------------------------------

The idea of this approach is to sample from the prior and then run the model. If the
log likelihood is then -inf, there is a problem, and the output of the theano functions is
inspected.

Sample from prior:

::

    from pymc3.util import (
        get_untransformed_name,
        is_transformed_name)

    varnames = list(map(str, model.vars))

    for name in varnames:
        if is_transformed_name(name):
            varnames.append(get_untransformed_name(name))

    with model:
        points = pm.sample_prior_predictive(var_names = varnames)
        points_list = []
        for i in range(len(next(iter(points.values())))):
            point_dict = {}
            for name, val in points.items():
                    point_dict[name] = val[i]
            points_list.append(point_dict)

points_list is a list of the starting points for the model, sampled from the prior.
Then to run the model and print the log-likelihood:

::

    fn = model.fn(model.logpt)

    for point in points_list[:]:
        print(fn(point))

To monitor the output and save it in a file (for use in ipython).
Learned from:
http://deeplearning.net/software/theano/tutorial/debug_faq.html#how-do-i-step-through-a-compiled-function

::

    %%capture cap --no-stderr
    def inspect_inputs(i, node, fn):
        print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs],
              end='')

    def inspect_outputs(i, node, fn):
        print(" output(s) value(s):", [output[0] for output in fn.outputs])

    fn_monitor = model.fn(model.logpt,
                          mode=theano.compile.MonitorMode(
                               pre_func=inspect_inputs,
                               post_func=inspect_outputs).excluding(
                                     'local_elemwise_fusion', 'inplace'))

    fn = model.fn(model.logpt)

    for point in points_list[:]:
        if fn(point) < -1e10:
            print(fn_monitor(point))
            break

In a new cell:

::

    with open('output.txt', 'w') as f:
        f.write(cap.stdout)

Then one can open output.txt in a text editor, and follow from where infs or nans come from
by following the inputs and outputs up through the graph


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