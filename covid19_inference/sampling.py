import random
import logging

import pymc3 as pm
import arviz as az
import numpy as np

log = logging.getLogger(__name__)


def get_start_points(trace, trace_az, frames_start=None, SD_chain_logl=2.5):
    r"""
    Returns the starting points such that the chains deviate at most SD_chain_logl
    standard deviations from the chain with the highest likelihood
    Parameters
    ----------
    trace : multitrace object
    trace_az : arviz trace object
    frames_start : int
        Which frames to use for calculating the mean likelihood and its standard deviation.
        By default it is set to the last third of the tuning samples
    SD_chain_logl : float
        The number of standard deviations. 2.5 as default
    Returns
    -------
    start_points :
        A list of starting points
    """
    logl = trace_az.warmup_sample_stats["lp"]
    n_tune = logl.shape[1]
    n_chains = logl.shape[0]
    if frames_start is None:
        frames_start = 3 * n_tune // 4
    logl_mean = np.array(logl[:, frames_start:].mean(axis=1))
    logl_std = np.array(logl[:, frames_start:].std(axis=1))
    max_idx = np.argmax(logl_mean)
    logl_thr = logl_mean[max_idx] - logl_std[max_idx] * SD_chain_logl
    keep_chains = logl_mean >= logl_thr
    log.info(f"Num chains kept: {np.sum(keep_chains)}/{n_chains}")

    start_points = []
    for i, keep_chain in enumerate(keep_chains):
        if keep_chain:
            start_points.append(trace.point(-1, chain=i))
    # order by logl:
    start_points = [
        p
        for _, p in sorted(
            zip(logl_mean[keep_chains], start_points), key=lambda pair: pair[0]
        )
    ]
    return start_points


def robust_sample(
    model, tune, draws, tuning_chains, final_chains, args_start_points=None, **kwargs
):
    r"""
    Samples the model by starting more chains than needed (tuning chains) and using only
    a reduced number final_chains for the final sampling.

    Parameters
    ----------
    model : :class:`Cov19Model`
        The model
    tune : int
        Number of tuning samples
    draws : int
        Number of final samples
    tuning_chains : int
        Number of tuning chains
    final_chains : int
        Number of draw chains
    args_start_points : dict
        Arguments passed to `get_start_points`
    **kwargs :
        Arguments passed to `pm.sample`

    Returns
    -------
    trace : trace as multitrace object
    trace_az : trace as arviz object

    """
    trace_tuning = pm.sample(
        model=model,
        tune=tune,
        draws=0,
        chains=tuning_chains,
        return_inferencedata=False,
        discard_tuned_samples=False,
        **kwargs,
    )
    trace_tuning_az = az.from_pymc3(trace_tuning, model=model, save_warmup=True)
    if args_start_points is None:
        args_start_points = {}
    start_points = get_start_points(trace_tuning, trace_tuning_az, **args_start_points)
    num_start_points = len(start_points)

    if num_start_points < final_chains:
        log.warning(
            "Not enough chains converged to minimum, we recommend increasing the number of tuning chains"
        )
        start_points += random.choices(start_points, final_chains - num_start_points)
    elif num_start_points > final_chains:
        start_points = start_points[:final_chains]

    trace = pm.sample(
        model=model,
        tune=tuning_chains // 3,
        draws=draws,
        chains=final_chains,
        start=start_points,
        return_inferencedata=False,
        discard_tuned_samples=False,
        **kwargs,
    )
    trace_az = az.from_pymc3(trace, model=model, save_warmup=True)
    return trace, trace_az
