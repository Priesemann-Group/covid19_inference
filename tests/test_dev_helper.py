import sys
import os

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))


def test_dev_helper():
    import pymc as pm
    import covid19_inference as cov
    import arviz as az

    model, trace = cov.create_example_instance()

    assert isinstance(model, pm.Model)
    assert isinstance(trace, az.InferenceData)
