# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-23 14:43:08
# @Last Modified: 2020-04-23 17:43:58
# ------------------------------------------------------------------------------ #


def test_dev_helper():
    import pymc3 as pm
    import theano
    import theano.tensor as tt
    import covid19_inference as cov

    model, trace = cov.create_example_instance()

    assert isinstance(model, pm.Model)
    assert isinstance(trace, pm.backends.base.MultiTrace)
