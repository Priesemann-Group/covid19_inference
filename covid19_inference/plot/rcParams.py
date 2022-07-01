colors = {
    "data": "#060434",  # black
    "fraction": "#33BBEE",  # cyan
    "cases": "#0077BB",  # teal
    "male": "#EE7733",  # orange
    "female": "#EE3377",  # magenta
    "Repr": "#009988",  # blue
    "c": "#e5f6ff2",
    "d": "#f2f4f8",
}


def get_rcparams_default():
    """
    Get a Param (dict) of the default parameters.
    Here we set our default values. Assigned once to module variable
    `rcParamsDefault` on load.
    """
    par = Param(
        locale="en_US",
        date_format="%b %d",  # Removed - in %-d because of windows...
        date_show_minor_ticks=True,
        rasterization_zorder=-1,
        draw_ci_95=True,
        draw_ci_75=False,
        draw_ci_50=False,
        color_model="#009988",
        color_data="#060434",
        color_prior="#708090",
        color_posterior="#009988",
        color_annot="#646464",
        color_male="#EE7733",
        color_female="#EE3377",
        color_championship_range="#c6e2f7",
        timeseries_xticklabel_rotation=0,  # In degree
    )

    return par


def set_rcparams(par):
    """
    Sets the rcparameters used for plotting, provided instance of `Param` has to have
    the following keys (attributes):

    Attributes
    ----------
    locale : str
        region settings, passed to `setlocale()`. Default: "en_US"

    date_format : str
        Format the date on the x axis of time-like data (see https://strftime.org/)
        example April 1 2020:
        "%m/%d" 04/01, "%-d. %B" 1. April
        Default "%b %-d", becomes April 1

    date_show_minor_ticks : bool
        Whether to show the minor ticks (for every day). Default: True

    rasterization_zorder : int or None
        Rasterizes plotted content below this value, set to None to keep everything
        a vector.
        |Default| -1

    draw_ci_95 : bool
        For time series plots, indicate 95% Confidence interval via fill between.
        |default| True

    draw_ci_75 : bool
        For time series plots, indicate 75% Confidence interval via fill between.
        |default| False

    draw_ci_50 : bool
        For time series plots, indicate 50% Confidence interval via fill between.
        |default| False

    color_model : str
        Base color used for model plots, mpl compatible color code "C0", "#303030"
        |default| "tab:green"

    color_data : str
        Base color used for data
        |default| "tab:blue"

    color_annot : str
        Color to use for annotations
        |default| "#646464"

    color_prior : str
        Color to used for priors in distributions
        |default| "#708090"

    color_posterior : str
            Color used in posterior plotting

    Example
    -------
    .. code-block:: python

        # Get default parameter
        pars = cov.plot.get_rcparams_default()

        # Change parameters
        pars["locale"]="de_DE"
        pars["color_data"]="tab:purple"

        # Set parameters
        cov.plot.set_rcparams(pars)
    ..
    """
    for key in get_rcparams_default().keys():
        assert key in par.keys(), "Provide all keys that are in .get_rcparams_default()"

    global rcParams
    rcParams = copy.deepcopy(par)


class Param(dict):
    """
    Paramters Base Class (a tweaked dict)

    We inherit from dict and also provide keys as attributes, mapped to `.get()` of
    dict. This avoids the KeyError: if getting parameters via `.the_parname`, we
    return None when the param does not exist.

    Avoid using keys that have the same name as class functions etc.

    Example
    -------
    .. code-block:: python

        foo = Param(lorem="ipsum")
        print(foo.lorem)
        >>> 'ipsum'
        print(foo.does_not_exist is None)
        >>> True
    ..
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return Param(copy.deepcopy(dict(self), memo=memo))

    @property
    def varnames(self):
        return [*self]


# ------------------------------------------------------------------------------ #
# init
# ------------------------------------------------------------------------------ #
# set global parameter variables
rcParams = get_rcparams_default()
