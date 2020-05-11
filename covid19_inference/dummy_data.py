import logging
import numpy as np
from numpy import exp
import datetime
from scipy.stats import halfcauchy, rv_discrete, nbinom
from scipy.special import binom, gammaln as gamln
import pandas as pd

log = logging.getLogger(__name__)


class DummyData(object):
    """
    Class for generating a random dataset, can be used for testing the model.

    Example
    -------

        .. code-block::

            #Create dates for data generation
            data_begin = datetime.datetime(2020,3,10)
            data_end = datetime.datetime(2020,4,26)

            # Create dummy data object
            dd = cov19.dummy_data.DummyData(data_begin,data_end)        

            #We can look at our initially generated values and tweak them by the attribute `dd.initials`.
            #If we are happy with the initial values we can generate our dummy data set.
            dd.generate()
            The generated data is accessible by the attribute `dd.data` as pandas dataframe.
    """

    def __init__(
        self,
        data_begin,
        data_end,
        mu=0.13,
        noise=False,
        noise_factor=0.00001,
        auto_generate=False,
        seed=None,
        **initial_values,
    ):
        """
        Creates a dummy dataset from initial values, these initial values get randomly
        generated or can be given via a dict.

        Parameters
        ----------
        data_begin : datetime.datetime
            Start date for the dataset
        data_end : datetime.datetime
            End date for the dataset
        mu : number
            Value for the recovery rate
        noise : bool
            Add random noise to the output
        noise_factor : number
            Alpha value for the negative binomial rn generator 
        auto_generate : bool
            Whether or not to generate a dataset on class init. Calls the self.generate() method.
        seed : number
            seed for the random number generation
        initial_values : dict
        """
        if seed is not None:
            np.random.seed(seed)

        self.noise = noise
        self.data_begin = data_begin
        self.data_end = data_end
        self.noise_factor = noise_factor
        self.mu = mu

        self._update_initial(**initial_values)

        if auto_generate:
            self.generate()

    @property
    def dates(self):
        return pd.date_range(self.data_begin, self.data_end)

    @property
    def data_len(self):
        return (self.data_end - self.data_begin).days

    @property
    def get_lambda_t(self):
        """Return lambda_t as dataframe with datetime index"""
        df = pd.DataFrame()
        df["date"] = self.dates
        df["lambda_t"] = self.lambda_t
        df = df.set_index("date")
        return df

    def _update_initial(self, **initial_values):
        # Generate inital values if they are not in the dict
        self.initials = {}

        if "S_initial" in initial_values:
            self.initials["S_initial"] = initial_values.get("S_initial")
        else:
            self.initials["S_initial"] = np.random.randint(5000000, 10000000)

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

    def _changepoints_to_lambda_t(self):
        """
        Generates a lambda_t array from the objects changepoints (self.changepoints).
        For now the lambda value jumps from one changepoint to the next.
        
        TODO
        ----
        Implement sigmoids/logistics 
        """

        # We create an lambda_t array for each date in our time period
        change_points = self.initials["change_points"]
        change_points.sort(key=lambda x: x[0])
        change_points = np.array(change_points)
        if self.data_begin == change_points[0, 0]:
            lambda_t = [change_points[0, 1]]
        else:
            lambda_t = [self.initials["lambda_initial"]]

        lambda_cp = self.initials["lambda_initial"]
        j = 0
        for i in range(self.data_len):
            if (self.data_begin + datetime.timedelta(days=i)) in change_points[:, 0]:
                lambda_cp = change_points[j][1]
                j = j + 1
            lambda_t.append(lambda_cp)

        return lambda_t

    def generate(self):
        """
        Generates a dataset from the initial values which were gives at initialization of the class
        or which were generated random. This generated data is saved to the `self.data` attribute.

        Returns
        -------
        dataset : pandas.dataFrame
        """

        # Create lambda_t array
        self.lambda_t = self._changepoints_to_lambda_t()

        # solves sir differential equation using rk4
        t_n, data = self._generate_SIR()

        # Add noise

        # Create cumulative cases
        if self.noise:
            cumulative_I = self._random_noise(
                self._create_cumulative(data[:, 1]), self.noise_factor
            )
            cumulative_R = self._random_noise(
                self._create_cumulative(data[:, 2]), self.noise_factor
            )
        else:
            cumulative_I = self._create_cumulative(data[:, 1])
            cumulative_R = self._create_cumulative(data[:, 2])

        # Construct pandas dataframe to return
        df = pd.DataFrame()
        df["date"] = pd.date_range(self.data_begin, self.data_end)
        df["confirmed"] = cumulative_I
        df["recovered"] = cumulative_R
        df["confirmed"] = df["confirmed"].astype(int)
        df["recovered"] = df["recovered"].astype(int)
        df = df.set_index("date")

        # shift the recovered and confirmed cases
        df = self._delay_cases(df)

        # Save as attribute and return
        self.data = df
        return df

    def _generate_SIR(self):
        r"""
        Numerically solves the differential equation

        .. math::
            \frac{dy}{dt}  = \lambda(t) \cdot
            \begin{bmatrix}
                -1 \\ 1 \\ 0 
            \end{bmatrix}
            \frac{y_0 \cdot y_1}{N} -
            \begin{bmatrix}
                0 \\ \mu y_1 \\ -\mu y_1
            \end{bmatrix}

        using runge kutta 4, whereby

        .. math::
            y = \begin{bmatrix}
                S \\ I \\ R 
            \end{bmatrix}
            = \begin{bmatrix}
                y_0 \\ y_1 \\ y_2 
            \end{bmatrix}

        and the population size :math:`N` is the sum of :math:`S`, :math:`I` and :math:`R`

        The initial SIR parameters and the force of infection :math:`\lambda(t)` are obtained from the object.
        
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
                [0, -self.mu * y[1], self.mu * y[1]]
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
        time_range = self.data_len
        N = (
            self.initials["S_initial"]
            + self.initials["I_initial"]
            + self.initials["R_initial"]
        )
        t_n = [0]
        y_n = [
            np.array(
                [
                    self.initials["S_initial"],
                    self.initials["I_initial"],
                    self.initials["R_initial"],
                ]
            )
        ]
        dt = 1

        # ------------------------------------------------------------------------------ #
        # Timesteps
        # ------------------------------------------------------------------------------ #
        for i in range(time_range):
            # Check if lambda_t has to change (initial value is also set by this)
            t_n.append(t_n[i] + dt)
            y_n.append(RK4(dt, t_n[i], y_n[i], self.lambda_t[i]))

        return t_n, np.array(y_n)

    def _create_cumulative(self, array):
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

    def _delay_cases(self, df):
        """
        Shifts each column of a given pandas dataframe by 10 days. This is done to simulate delayed cases.
        
        Parameters
        ----------
        df : pandas.dataFrame

        Returns
        -------
        df : pandas.dataFrame
            delayed cases

        TODO
        ----
        Look into this a bit more
        """
        delta = datetime.timedelta(days=10)
        df = df.shift(periods=10).dropna()
        return df

    def _random_noise(self, array, factor):
        r"""
        Generates random noise on an observable by a Negative Binomial :math:`NB`.
        References to the negative binomial can be found `here <https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Negative_Binomial_Regression.pdf>`_
        .

        .. math::
            O &\sim NB(\mu=datapoint,\alpha=0.0001)
        
        We keep the alpha parameter low to obtain a small variance which should than always be approximatly the size of the mean.

        Parameters
        ----------
        array : 1-dim
            observable on which we want to add the noise

        factor : number


        Returns
        -------
        array : 1-dim
            observable with added noise
        """

        def convert(mu, alpha):
            r = 1 / alpha
            p = mu / (mu + r)
            return r, 1 - p

        for i in range(len(array)):
            if array[i] == 0:
                continue
            log.debug(f"Data {array[i]}")
            r, p = convert(array[i], factor)
            log.debug(f"n {r}, p {p}")
            mean, var = nbinom.stats(r, p, moments="mv")
            log.debug(f"mean {mean} var {var}")
            array[i] = nbinom.rvs(r, p)
            log.debug(f"Drawn {array[i]}")

        return array
