import logging
import numpy as np
import datetime
from scipy.stats import nbinom, halfcauchy
import pandas as pd

log = logging.getLogger(__name__)


def get_dummy_data(data_begin, data_end, lambda_initial):
    r"""
    Generates a random dataset by drawing random initial values and random changepoints.
    By numericaly solving the SIR differential equation with these initial values a dataset is created.
    Finally a noise term is added onto the dataset.

    Parameters
    ----------
    data_begin, data_end : datetime.datetime
        Range for the random date

    lambda_initial:
        initial lambda value

    Returns
    -------
    df, changepoints : pandas.DataFrame,array

  """

    # Draw initial SIR parameters
    S_initial = np.random.randint(500, 100000)  # Population in germany as upper bound
    I_initial = int(halfcauchy.rvs(loc=4))  # Offset by one to get atleast one
    R_initial = 0

    log.debug(f"Initial susceptible {S_initial}")
    log.info(f"Initial infected {I_initial}")
    # Draw changepoints every weekend
    changepoints = []
    lambda_t = lambda_initial

    for date in pd.date_range(data_begin, data_end):
        if date.weekday() == 6:
            # Draw a date
            date_index = int(np.random.normal((date - data_begin).days, 3))
            date_random = data_begin + datetime.timedelta(days=date_index)
            # Draw a lambda
            lambda_t = np.random.normal(lambda_initial, lambda_initial / 2)
            changepoints.append([date_random, lambda_t])

    # Generate SIR data from the random changepoints
    t_n, data = _generate_SIR(
        data_begin,
        data_end,
        [S_initial, I_initial, R_initial],
        0.4,
        lambda_initial,
        changepoints,
    )
    cumulative_I = _create_cumulative(data[:, 1])
    cumulative_R = _create_cumulative(data[:, 2])

    df = pd.DataFrame()
    df["date"] = pd.date_range(data_begin, data_end)

    df["confirmed"] = _random_noise(cumulative_I)
    df["recovered"] = _random_noise(cumulative_R)

    df["confirmed"] = df["confirmed"].astype(int)
    df["recovered"] = df["recovered"].astype(int)
    df = df.set_index("date")
    return df, changepoints


def _create_cumulative(array):
    r"""
    Since we cant measure a negative number of new cases. All the negative values are cut from the I/R array
    and the counts are added up to create the cumulative/total cases dataset.
    """
    # Confirmed
    diff = [array[0]]
    for i in range(1, len(array)):
        if (array[i] - array[i - 1]) > 0:
            diff.append(array[i] - array[i - 1] + diff[i - 1])
        else:
            diff.append(diff[i - 1])

    return diff


def _generate_SIR(data_begin, data_end, SIR_initial, mu, lambda_init, changepoints):
    r"""
    Generates a dummy dataset for the SIR model using the vector 

    .. math::
        y = \begin{bmatrix}
            S \\ I \\ R 
        \end{bmatrix}
        = \begin{bmatrix}
            y_0 \\ y_1 \\ y_2 
        \end{bmatrix}

    The differential euqation

    .. math::
        \frac{dy}{dt}  = \lambda(t) \cdot
        \begin{bmatrix}
            -1 \\ 1 \\ 0 
        \end{bmatrix}
        \frac{y_0 \cdot y_1}{N} -
        \begin{bmatrix}
            0 \\ \mu y_1 \\ -\mu y_1
        \end{bmatrix}
    
    gets solved numerically for the initial SIR parameters using runge kutta.
    Additionally the force of infection :math:`\lambda(t)` can be supplied using the changepoints
    parameter. The population size :math:`N` is the sum over :math:`S`, :math:`I` and :math:`R`.

    Parameters
    ----------
    data_begin : datetime.datetime
        date at which the dummy data begins
    data_end : datetime.datetime
        end date for the dummy data
    SIR_initial : np.array [S_initial,I_initial,R_initial]
        starting values for the numerical integration, population gets calculated by S+I+R
    mu : number
        recovery rate
    changepoints : list
        should contain the lambda_t value and the corresponding date, in decending order
        ([date_1,lambda_1 ,],[lambda_2, date_2])

    Return
    ------
    dataframe : pandas.DataFrame

    """

    # SIR model as vector
    def f(t, y, lambda_t):
        return lambda_t * np.array([-1.0, 1.0, 0.0]) * y[0] * y[1] / N + np.array(
            [0, -mu * y[1], mu * y[1]]
        )

    # Runge_Kutta_4 timesteps
    def RK4(dt, t_n, y_n, lambda_t):
        k_1 = f(t_n, y_n, lambda_t)
        k_2 = f(t_n + dt / 2, y_n + k_1 * dt / 2, lambda_t)
        k_3 = f(t_n + dt / 2, y_n + k_2 * dt / 2, lambda_t)
        k_4 = f(t_n + dt, y_n + k_3 * dt, lambda_t)
        return y_n + dt / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)

    # ------------------------------------------------------------------------------ #
    # Preliminar parameters
    # ------------------------------------------------------------------------------ #
    time_range = (data_end - data_begin).days
    N = SIR_initial[0] + SIR_initial[1] + SIR_initial[2]
    t_n = [0]
    y_n = [np.array(SIR_initial)]
    dt = 1
    lambda_t = lambda_init
    # Changepoints
    changepoints.sort(key=lambda x: x[0])  # Sort by date (increasing)
    j = 0
    for i in range(time_range):
        # Check if lambda_t has to change (initial value is also set by this)
        if (data_begin + datetime.timedelta(days=i)) in changepoints[:][0]:
            lambda_t = changepoints[j][1]
            j = j + 1
        t_n.append(t_n[i] + dt)
        y_n.append(RK4(dt, t_n[i], y_n[i], lambda_t))

    return t_n, np.array(y_n)


def _random_noise(array):
    r"""
    Generates random noise on an observable by a Negative Binomial :math:`NB`.

    .. math::
        O &\sim NB(\mu=data,\theta=10)
        
    Parameters
    ----------
    array : 1-dim
        observable on which we want to add the noise

    Returns
    -------
    array : 1-dim
        observable with added noise
    """

    def convert_params(mu, theta):
        """
        Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

        See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
        """
        r = theta
        var = mu + 1 / r * mu ** 2
        p = (var - mu) / var
        return r, 1 - p

    for data in array:
        if data == 0:
            continue
        n, p = convert_params(data, 10)
        # mean, var, skew, kurt = nbinom.stats(n=n, p=p, moments='mvsk', )
        # log.debug(f"Mean {mean}, var {var}, data {data}")
        data = nbinom.rvs(n=n, p=p)
        log.debug(f"Drawn {data}")
    return array
