# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-04-23 14:43:08
# @Last Modified: 2020-04-23 17:22:17
# ------------------------------------------------------------------------------ #

def test_dev_helper():
    import pymc3 as pm
    import covid19_inference as cov

    model, trace = cov.create_example_instance()

    assert isinstance(model, pm.Model)
    assert isinstance(trace, pm.backends.base.MultiTrace)
