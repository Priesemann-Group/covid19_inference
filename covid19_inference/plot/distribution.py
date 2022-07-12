import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy import stats

from .rcParams import *

log = logging.getLogger(__name__)


def distribution(
    model,
    idata,
    key,
    nSamples_prior=1000,
    title="",
    dist_math="x",
    indices=None,
    ax=None,
):
    """
    High level plotting function for distribution overviews.
    Only works if the distrubtion is one dim or two dimensional.

    Parameters
    ----------
    model : Cov19Model
        The model used to create the inference data
    idata: av.InferenceData
        The inference data containing the posterior samples
    key: str
        The variable of interest (which should be plotted)
    nSamples_prior: int
        Number of samples to draw for the prior kernel density estimation.
    indices : array-like int
        Which dimensions do you want to plot from the variable? default: None i.e. all
    """

    data = idata.posterior[key]

    # This function all may change in a future pymc3 version
    try:
        prior = pm.sample_prior_predictive(
            samples=nSamples_prior, model=model, var_names=[key]
        ).prior[key]
        prior = np.array(prior).reshape(
            (prior.shape[0] * prior.shape[1],) + prior.shape[2:]
        )
    except ValueError:
        log.warning(f"Could not calculate prior for {key}")
        prior = None

    # Chain, draw, ...
    if data.ndim == 4:
        if ax is None:
            fig, axes = plt.subplots(
                indices.shape[0],
                indices.shape[1],
                figsize=(4 * indices.shape[1], 4 * indices.shape[0]),
            )
        else:
            axes = ax
            fig = ax[0, 0].get_figure()
        # Flatten chain and sample dimension
        data = np.array(data).reshape((data.shape[0] * data.shape[1],) + data.shape[2:])

        if indices is None:
            indices = (data.shape[-2], data.shape[1])

        for i in range(indices[0]):
            for j in range(indices[1]):
                _distribution(
                    array_posterior=data[:, i, j],
                    array_prior=prior[:, i, j] if prior is not None else None,
                    dist_name=title,
                    dist_math=dist_math,
                    suffix=f"{i,j}",
                    ax=axes[i, j] if hasattr(axes, "__len__") else axes,
                )
        if title:
            fig.suptitle(title)
        return axes
    elif data.ndim == 3:
        if indices is None:
            indices = [i for i in range(data.shape[-1])]
        if ax is None:
            fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
        else:
            axes = ax

        # Flatten chain and sample dimension
        data = np.array(data).reshape((data.shape[0] * data.shape[1],) + data.shape[2:])
        for j, i in enumerate(indices):
            _distribution(
                array_posterior=data[:, i],
                array_prior=prior[:, i] if prior is not None else None,
                dist_name=title,
                dist_math=dist_math,
                suffix=f"{i}" if len(indices) > 1 else "",
                ax=axes[j] if hasattr(axes, "__len__") else axes,
            )
        if title:
            fig.suptitle(title)
        return axes
    elif data.ndim == 2:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

        # Flatten chain and sample dimension
        data = np.array(data).reshape((data.shape[0] * data.shape[1]))
        _distribution(
            array_posterior=data,
            array_prior=prior,
            dist_name=title,
            dist_math=dist_math,
            ax=ax,
        )
        ax.set_title(title)
        return [ax]
    else:
        print("Dimension of data not supported!")


# ------------------------------------------------------------------------------ #
# Low level functions (copied from npis europe project)
# ------------------------------------------------------------------------------ #


def _distribution(
    array_posterior, array_prior, dist_name, dist_math, suffix="", ax=None
):
    """
    Low level function to plots posterior and prior from arrays.

    Parameters
    ----------
    array_posterior, array_prior : array or None
        Sampling data as array, should be filtered beforehand. If none
        it does not get plotted!

    dist_name: str
        name of distribution for plotting
    dist_math: str
        math of distribution for plotting

    suffix: str,optional
        Suffix for the plot title e.g. "age_group_1"
        Default: ""

    ax : mpl axes element, optional
        Plot into an existing axes element
        Default: :code:`None`


    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(4.5 / 3, 1),
        )

    # ------------------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------------------ #
    if array_posterior is not None:
        ax = _plot_posterior(x=array_posterior, ax=ax)
    if array_prior is not None:
        ax = _plot_prior(x=array_prior, ax=ax)

    # ------------------------------------------------------------------------------ #
    # Annotations
    # ------------------------------------------------------------------------------ #
    # add the overlay with median and CI values. these are two strings
    text_md, text_ci = _string_median_CI(array_posterior, prec=2)
    if suffix == "":
        text_md = f"${dist_math}={text_md}$"
    else:
        # Hack around double underscores in latex
        if "_" in dist_math:
            if "^" in dist_math:
                text_md = f"${dist_math}={text_md}$"
            else:
                text_md = f"${dist_math}^{{{suffix}}}={text_md}$"
        else:
            text_md = f"${dist_math}_{{{suffix}}}={text_md}$"

    # create the inset text elements, and we want a bounding box around the compound
    try:
        tel_md = ax.text(
            0.6,
            0.9,
            text_md,
            fontsize=12,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            zorder=100,
        )
        x_min, x_max, y_min, y_max = _get_mpl_text_coordinates(tel_md, ax)
        tel_ci = ax.text(
            0.6,
            y_min * 0.9,  # let's have a ten perecent margin or so
            text_ci,
            fontsize=9,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            zorder=101,
        )
        _add_mpl_rect_around_text(
            [tel_md, tel_ci],
            ax,
            facecolor="white",
            alpha=0.5,
            zorder=99,
        )
    except Exception as e:
        log.debug(f"Unable to create inset with {dist_name} value: {e}")

    # ------------------------------------------------------------------------------ #
    # Additional plotting settings
    # ------------------------------------------------------------------------------ #
    ax.xaxis.set_label_position("top")
    # ax.set_xlabel(dist["name"] + suffix)

    ax.tick_params(labelleft=False)
    ax.set_rasterization_zorder(rcParams.rasterization_zorder)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return ax


def _plot_prior(x, ax=None, **kwargs):
    """
    Low level plotting function, plots the prior as line for sampling data by using kernel density estimation.
    For more references see `scipy documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html>`_.

    It is highly recommended to pass an axis otherwise the xlim may be a bit wonky.

    Parameters
    ----------
    x :
        Input values, from sampling

    ax : mpl axes element, optional
        Plot into an existing axes element
        |default| :code:`None`

    kwargs : dict, optional
        Directly passed to plotting mpl.
    """
    reset_xlim = False
    if ax is None:
        fig, ax = plt.subplots()
        xlim = [x.min(), x.max()]
    else:
        # may need to convert axes values, and restore xlimits after adding prior
        xlim = ax.get_xlim()
        reset_xlim = True
    try:
        prior = stats.kde.gaussian_kde(
            x,
        )
    except Exception as e:  # Probably only one value of x
        log.warning(f"Could not plot prior! {x}")
        return ax
    x_for_ax = np.linspace(*xlim, num=1000)

    x_for_pr = x_for_ax

    ax.plot(
        x_for_ax,
        prior(x_for_ax),
        label="Prior",
        color=rcParams.color_prior,
        linewidth=3,
        **kwargs,
    )

    if reset_xlim:
        ax.set_xlim(*xlim)

    return ax


def _plot_posterior(x, bins=50, ax=None, **kwargs):
    """
    Low level plotting function to plot an sampling data as histogram.

    Parameters
    ----------
    x:
        Input values, from sampling

    bins: int, optional
        Defines the number of equal-width bins in the range.
        |default| 50

    ax : mpl axes element, optional
        Plot into an existing axes element
        |default| :code:`None`

    kwargs : dict, optional
        Directly passed to plotting mpl.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.hist(
        x,
        bins=bins,
        color=rcParams.color_posterior,
        label="Posterior",
        alpha=0.7,
        zorder=-5,
        density=True,
        **kwargs,
    )

    return ax


# ------------------------------------------------------------------------------ #
# Formating and util
# ------------------------------------------------------------------------------ #


def _string_median_CI(arr, prec=2):
    f_trunc = lambda n: _truncate_number(n, prec)
    med = f_trunc(np.median(arr))
    perc1, perc2 = (
        f_trunc(np.percentile(arr, q=2.5)),
        f_trunc(np.percentile(arr, q=97.5)),
    )
    # return "Median: {}\nCI: [{}, {}]".format(med, perc1, perc2)
    return f"{med}", f"[{perc1}, {perc2}]"


def _truncate_number(number, precision):
    return "{{:.{}f}}".format(precision).format(number)


def _get_mpl_text_coordinates(text, ax):
    """
    helper to get coordinates of a text object in the coordinates of the
    axes element [0,1].
    used for the rectangle backdrop.

    Returns:
    x_min, x_max, y_min, y_max
    """
    fig = ax.get_figure()

    try:
        fig.canvas.renderer
    except Exception as e:
        log.debug(e)
        # otherwise no renderer, needed for text position calculation
        fig.canvas.draw()

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    # get bounding box of text
    transform = ax.transAxes.inverted()
    try:
        bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    except:
        bb = text.get_window_extent()
    bb = bb.transformed(transform)
    x_min = bb.get_points()[0][0]
    x_max = bb.get_points()[1][0]
    y_min = bb.get_points()[0][1]
    y_max = bb.get_points()[1][1]

    return x_min, x_max, y_min, y_max


def _add_mpl_rect_around_text(text_list, ax, x_padding=0.05, y_padding=0.05, **kwargs):
    """
    add a rectangle to the axes (behind the text)

    provide a list of text elements and possible options passed to
    mpl.patches.Rectangle
    e.g.
    facecolor="grey",
    alpha=0.2,
    zorder=99,
    """

    x_gmin = 1
    y_gmin = 1
    x_gmax = 0
    y_gmax = 0

    for text in text_list:
        x_min, x_max, y_min, y_max = _get_mpl_text_coordinates(text, ax)
        if x_min < x_gmin:
            x_gmin = x_min
        if y_min < y_gmin:
            y_gmin = y_min
        if x_max > x_gmax:
            x_gmax = x_max
        if y_max > y_gmax:
            y_gmax = y_max

    # coords between 0 and 1 (relative to axes) add 10% margin
    y_gmin = np.clip(y_gmin - y_padding, 0, 1)
    y_gmax = np.clip(y_gmax + y_padding, 0, 1)
    x_gmin = np.clip(x_gmin - x_padding, 0, 1)
    x_gmax = np.clip(x_gmax + x_padding, 0, 1)

    rect = mpl.patches.Rectangle(
        (x_gmin, y_gmin),
        x_gmax - x_gmin,
        y_gmax - y_gmin,
        transform=ax.transAxes,
        **kwargs,
    )

    ax.add_patch(rect)
