import logging
import numpy as np
from numpy import exp
import datetime
from scipy.stats import halfcauchy, rv_discrete, nbinom
from scipy.special import binom, gammaln as gamln
import pandas as pd

log = logging.getLogger(__name__)


class DummyData(object):
    """docstring for DummyData"""

    def __init__(
        self, data_begin, data_end, mu=0.13, auto_generate=False, **initial_values
    ):
        """
        Creates a dummy dataset from initial values, these initial values get randomly
        generated or can be given via a dict.

        Parameters
        ----------
        data_begin, data_end : datetime.datetime
            Time range for the dataset
        auto_generate : bool
            Whether or not to generate a dataset on class init. Calls the self.generate() method.
        initial_values : dict

        """

        self.data_begin = data_begin
        self.data_end = data_end
        self._update_initial(**initial_values)

        if auto_generate:
            self.generate()

    @property
    def dates(self):
        return pd.date_range(self.data_begin, self.data_end)

    @property
    def data_len(self):
        return (self.data_end - self.data_begin).days

    def _update_initial(self, **initial_values):
        # Generate inital values if they are not in the dict
        self.initials = {}

        if "S_initial" in initial_values:
            self.initials["S_initial"] = initial_values.get("S_initial")
        else:
            self.initials["S_initial"] = np.random.randint(50000, 100000)

        if "I_initial" in initial_values:
            self.initials["I_initial"] = initial_values.get("I_initial")
        else:
            self.initials["I_initial"] = int(halfcauchy.rvs(loc=3))

        if "R_initial" in initial_values:
            self.initials["R_initial"] = initial_values.get("R_initial")
        else:
            self.initials["R_initial"] = 0

        if "lambda_initial" in initial_values:
            self.initials["lambda_initial"] = initial_values.get("lambda_initial")
        else:
            self.initials["lambda_initial"] = np.random.uniform(0, 1)

        if "change_points" in initial_values:
            self.initials["change_points"] = initial_values.get("change_points")
        else:
            self.initials["change_points"] = self._generate_change_points_we()

    def _generate_change_points_we(self):
        """
        Generates random change points each weekend in the give time period.

        The date is normal distributed with mean on each Saturday and a variance of 3 days.

        The lambda is draw uniform between 0 and 1
        """
        change_points = []
        for date in pd.date_range(self.data_begin, self.data_end):
            if date.weekday() == 6:
                # Draw a date
                date_index = int(np.random.normal(0, 3))
                date_random = date + datetime.timedelta(days=date_index)
                # Draw a lambda
                lambda_t = np.random.uniform(0, 1)
                change_points.append([date_random, lambda_t])
        return change_points

    def generate(self):

        # ------------------------------------------------------------------------------ #
        # Generate lambda(t) array
        # ------------------------------------------------------------------------------ #
        # From our lambda_t array we model the changepoints by logistics function whereby
        # the initial changepoints are the max/min values
        change_points = self.initials["change_points"]
        change_points.sort(key=lambda x: x[0])
        # We create an lambda_t array for each date in our time period

        lambda_t = [self.initials["lambda_initial"]]
        lambda_cp = self.initials["lambda_initial"]
        change_points = np.array(change_points)
        j = 0
        for i in range(time_range):
            if (data_begin + datetime.timedelta(days=i)) in change_points[:, 0]:
                lambda_cp = change_points[j][1]
                j = j + 1
            lambda_t.append(lambda_cp)

        print(lambda_t)

        """
        # Generate SIR data from the random changepoints
        t_n, data, lambda_t = self._generate_SIR(
                                self.data_begin,
                                self.data_end,
                                [self.initials["S_initial"], self.initials["I_initial"], self.initials["R_initial"],
                                self.initials["mu"],
                                self.initials[]])
        """

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

        # Changepoints
        lambda_t = [lambda_init]
        lambda_cp = lambda_init
        changepoints.sort(key=lambda x: x[0])  # Sort by date (increasing)
        changepoints = np.array(changepoints)
        j = 0
        for i in range(time_range):
            if (data_begin + datetime.timedelta(days=i)) in changepoints[:, 0]:
                lambda_cp = changepoints[j][1]
                j = j + 1
            lambda_t.append(lambda_cp)

        # ------------------------------------------------------------------------------ #
        # Timesteps
        # ------------------------------------------------------------------------------ #
        for i in range(time_range):
            # Check if lambda_t has to change (initial value is also set by this)
            t_n.append(t_n[i] + dt)
            y_n.append(RK4(dt, t_n[i], y_n[i], lambda_t[i]))

        return t_n, np.array(y_n), lambda_t
