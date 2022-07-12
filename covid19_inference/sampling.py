import random
import logging
import warnings
from collections import Counter
import pickle
import glob
import os
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np


log = logging.getLogger(__name__)


def get_start_points(trace, trace_az, frames_start=None, SD_chain_logl=2.5):
    r"""
    Returns the starting points such that the chains deviate at most SD_chain_logl
    standard deviations from the chain with the highest likelihood.
    Parameters
    ----------
    trace : multitrace object
    trace_az : arviz trace object
    frames_start : int
        Which frames to use for calculating the mean likelihood and its standard deviation.
        By default it is set to the last third of the tuning samples
    SD_chain_logl : None or float
        The number of standard deviations. 2.5 as default. If None, keep all chains
    Returns
    -------
    start_points :
        A list of starting points
    logl_mean :
        The mean log-likelihood of the starting points
    """
    logl = trace_az.warmup_sample_stats["lp"]
    n_tune = logl.shape[1]
    n_chains = logl.shape[0]
    if frames_start is None:
        frames_start = 3 * n_tune // 4
    logl_mean = np.array(logl[:, frames_start:].mean(axis=1))
    logl_std = np.array(logl[:, frames_start:].std(axis=1))
    max_idx = np.argmax(logl_mean)
    if SD_chain_logl is not None:
        logl_thr = logl_mean[max_idx] - logl_std[max_idx] * SD_chain_logl
        keep_chains = logl_mean >= logl_thr
        log.info(f"Num chains kept: {np.sum(keep_chains)}/{n_chains}")
    else:
        keep_chains = np.ones_like(logl_mean).astype("bool")

    start_points = []
    for i, keep_chain in enumerate(keep_chains):
        if keep_chain:
            start_points.append(trace.point(-1, chain=i))

    return start_points, logl_mean[keep_chains]


class Callback:
    """
    Simple callback to save the trace every n iterations and
    plot the logp.

    Parameters
    ----------
    path : str
        Path to save the trace

    name : str
        Name of the model, should be used when running multiple
        models in parallel (default: "model")

    n : int
        Save the trace every n iterations
    """

    def __init__(self, path="/temp", name="model", n=100):
        self.path = path
        self.name = name
        self.n = n
        self.lengths = Counter()

        # Setup plotting of logp
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 5))
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Logp")
        self.ax.set_title(name)

    """This function is called by pymc3 every iterations
    """

    def __call__(self, trace, draw):

        # Update values
        self.lengths[draw.chain] += 1

        # Save the trace every n iterations
        if self.lengths[draw.chain] % self.n == 0:
            self.save(trace, draw.chain)
            self.plot_logp(trace, draw.chain)

    def plot_logp(self, trace, chain):
        pm_trace = pm.backends.base.MultiTrace({chain: trace}.values())
        masked_logp = pm_trace["model_logp"][pm_trace["model_logp"] != 0]
        try:
            self.ax.plot(
                np.arange(self.lengths[chain] - len(masked_logp), self.lengths[chain]),
                masked_logp,
            )
            self.fig.savefig(os.path.join(self.path, self.name) + "_logp.png")
        except:
            log.warning(f"Could not save {os.path.join(self.path, self.name)}_logp.png")

    def save(self, trace, chain):
        try:
            with open(os.path.join(self.path, self.name) + f"_{chain}.pkl", "wb") as f:
                trace.chain = chain
                pickle.dump(trace, f)
        except:
            log.warning(
                f"Could not save {os.path.join(self.path, self.name)}_{chain}.pkl"
            )

    def load_all(self):
        files = glob.glob(f"{self.path}/{self.name}_*.pkl")
        traces = {}
        for f in files:
            with open(f, "rb") as f:
                trace = pickle.load(f)
                traces[trace.chain] = trace
        return az.from_pymc3(pm.backends.base.MultiTrace(traces.values()))


def burn_in(
    model,
    n_tune,
    n_chains_burn_in,
    n_chains_final,
    step_method,
    start_points=None,
    callback=None,
    args_start_points=None,
    sample_kwargs=None,
):
    if sample_kwargs is None:
        sample_kwargs = {}
    i = 0
    while i < 50:
        try:
            trace_tuning = pm.sample(
                model=model,
                tune=n_tune,
                draws=0,
                start=start_points,
                chains=n_chains_burn_in,
                return_inferencedata=False,
                discard_tuned_samples=False,
                step=step_method,
                callback=callback,
                **sample_kwargs,
            )
        except RuntimeError as error:
            if i < 10:
                i += 1
                log.warning(
                    f"Tuning lead to a nan error in one chain, "
                    f"trying again (try no {i})."
                )
                continue

            else:
                raise error
        i = 1000

    trace_tuning_az = az.from_pymc3(trace_tuning, model=model, save_warmup=True)
    if args_start_points is None:
        args_start_points = {}
    start_points, logl_starting_points = get_start_points(
        trace_tuning, trace_tuning_az, SD_chain_logl=None, **args_start_points
    )
    num_start_points = len(start_points)

    if num_start_points < n_chains_final:
        log.warning(
            "Not enough chains converged to minimum, we recommend increasing the number of tuning chains"
        )
        start_points = random.choices(start_points, k=n_chains_final)
    elif num_start_points > n_chains_final:
        p = np.exp(logl_starting_points - max(logl_starting_points))
        start_points = np.random.choice(
            start_points,
            size=n_chains_final,
            p=p / np.sum(p),
            replace=False,
        )
    return start_points, trace_tuning_az


def robust_sample(
    model,
    tune,
    draws,
    final_chains,
    burnin_chains,
    burnin_draws=None,
    burnin_chains_2nd=None,
    burnin_draws_2nd=None,
    args_start_points=None,
    callback=None,
    sample_kwargs=None,
    **kwargs,
):
    r"""
    Samples the model by starting more chains than needed (burn-in chains) and using only
    a reduced number final_chains for the final sampling. The final chains are randomly
    chosen (without replacement) weighted by their likelihood.
    Parameters
    ----------
    model : :class:`Cov19Model`
        The model
    tune : int
        Number of tuning samples
    draws : int
        Number of final samples
    final_chains : int
        Number of draw chains
    burnin_chains : int
        Number of chains used during burn-in, recommended to use about 2-3 time more than
        the number of final_chains
    burnin_draws : int
        Length of the burn-in period, can be fairly short, on the order of a few hundreds
        draws. By default it set to tune//2
    burnin_chains_2nd : int
        If not None, use a two-stage burn-in period, reducing the number of chains each time,
        Therefore, it should be less than burnin_chains and more than final_chains:
        burnin_chains > burnin_chains_2nd > final_chains
    burnin_draws_2nd : int
        Length of the second burn-in period. By default it set burnin_draws
    args_start_points : dict
        Arguments passed to `get_start_points`
    tune_2nd : int
        If set, use different number of tuning samples for the second tuning
    sample_kwargs:
        Arguments passed to pm.sample
    **kwargs :
        Arguments passed to the nuts step function.

    Returns
    -------
    trace : trace as multitrace object
    trace_az : trace as arviz object

    """

    burnin_chains_2nd_to_compare = (
        burnin_chains_2nd if burnin_chains_2nd is not None else final_chains + 0.5
    )
    if not burnin_chains > burnin_chains_2nd_to_compare > final_chains:
        raise RuntimeError(
            "The number of chains should decrease for good sampline: "
            "burnin_chains > burnin_chains_2nd > final_chains"
        )

    if burnin_draws is None:
        burnin_draws = tune // 2
    if burnin_draws_2nd is None and burnin_chains_2nd is not None:
        burnin_draws_2nd = burnin_draws

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*invalid value encountered in double_scalars.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*Tuning samples will be included in the returned.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*Tuning was enabled throughout the whole trace.*"
        )
        warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
        warnings.filterwarnings(
            "ignore",
            message=".*The number of samples is too small to check convergence reliably.*",
        )

        # Create nuts step method class to reuse for tuning and later sampling
        default_nuts = pm.NUTS(model=model, **kwargs)
        if sample_kwargs is None:
            sample_kwargs = {}

        ## Burn-in
        start_points, trace_burn_in_1_az = burn_in(
            model,
            n_tune=burnin_draws,
            n_chains_burn_in=burnin_chains,
            n_chains_final=final_chains
            if burnin_chains_2nd is None
            else burnin_chains_2nd,
            step_method=default_nuts,
            args_start_points=args_start_points,
            sample_kwargs=sample_kwargs,
        )
        if burnin_chains_2nd is not None:
            start_points, trace_burn_in_2_az = burn_in(
                model,
                n_tune=burnin_draws_2nd,
                n_chains_burn_in=burnin_chains_2nd,
                n_chains_final=final_chains,
                start_points=start_points,
                step_method=default_nuts,
                args_start_points=args_start_points,
                sample_kwargs=sample_kwargs,
            )

        trace = pm.sample(
            model=model,
            tune=tune,
            draws=draws,
            chains=final_chains,
            start=start_points,
            return_inferencedata=False,
            discard_tuned_samples=False,
            step=default_nuts,
            callback=callback,
            **sample_kwargs,
        )
        trace_az = az.from_pymc3(trace, model=model, save_warmup=True)

        def append_burn_in_samples(trace, trace_burn_in, num):
            new_names = [
                f"burn-in-{num}_" + n.replace("warmup_", "")
                for n in trace_burn_in.groups()
            ]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The group .* is not defined in the .* scheme",
                )
                trace.add_groups(
                    {n: o for n, o in zip(new_names, trace_burn_in.values())}
                )
            return trace

        trace_az = append_burn_in_samples(trace_az, trace_burn_in_1_az, 1)
        if burnin_chains_2nd is not None:
            trace_az = append_burn_in_samples(trace_az, trace_burn_in_2_az, 2)

    return trace, trace_az
