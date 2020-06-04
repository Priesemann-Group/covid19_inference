# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-05-26 12:32:52
# @Last Modified: 2020-06-04 14:57:27
# ------------------------------------------------------------------------------ #
import logging
import numpy as np

from scipy.stats import halfcauchy, rv_discrete, nbinom
import pandas as pd
import datetime
log = logging.getLogger(__name__)

#Dirty Hack to hold the model ctx
ctx = []

class DummyModel(object):
    """docstring for DummyModel"""
    def __init__(
        self,
        data_begin,
        data_end,
        mu=0.13,
        dt = 0.001,
        seed=None,
        **initial_values
    ):
        """
            Base class for the dummy data model.
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
            auto_generate : bool
                Whether or not to generate a dataset on class init. Calls the :py:meth:`generate` method.
            seed : number
                Seed for the random number generation. Also accessible by `self.seed` later on. 
            initial_values : dict
                Kwargs for most of the initial values see :py:meth:`_update_initial`
        """
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(1000000,9999999)
        np.random.seed(self.seed)

        self._data_begin = data_begin
        self._data_end = data_end
        self._dt = dt
        self.mu = mu

        #Generate initial values
        self._initial(**initial_values)

        #Append object (context) to a list (dirty hack to get easily later)
        ctx.append(self)



    # ------------------------------------------------------------------------------ #
    # Data dates
    # ------------------------------------------------------------------------------ #

    @property
    def dates(self):
        return pd.date_range(self.data_begin, self.data_end)

    @property
    def data_begin(self):
        return self._data_begin
    
    @property
    def data_end(self):
        return self._data_end
    
    @property
    def data_len(self):
        return (self.data_end - self.data_begin).days


    # ------------------------------------------------------------------------------ #
    # Runge kutta 
    # ------------------------------------------------------------------------------ #
    @property
    def dt(self):
        return self._dt
    

    # ------------------------------------------------------------------------------ #
    # Inital values
    # ------------------------------------------------------------------------------ #

    def _initial(self, **inp_init):
        """
            Generates the initial values with the seed given at :py:meth:`__init__`.

            Parameters
            ----------
            
        """        

        def _generate_compartmental():
            """
            Genrate initial values for compartmental models
            """
            # ------------------------------------------------------------------------------ #
            # SIR model
            # ------------------------------------------------------------------------------ #
            self.initials["SIR"] = dict()
            if "S" in inp_init:
                self.initials["SIR"]["S"] = inp_init.get("S")
            else:
                self.initials["SIR"]["S"] = 1000        

            if "I" in inp_init:
                self.initials["SIR"]["I"] = inp_init.get("I")
            else:
                self.initials["SIR"]["I"] = int(halfcauchy.rvs(loc=3))

            if "R" in inp_init:
                self.initials["SIR"]["R"] = inp_init.get("R")
            else:
                self.initials["SIR"]["R"] = 0

            # ------------------------------------------------------------------------------ #
            # SEIR model
            # ------------------------------------------------------------------------------ #
            self.initials["SEIR"] = dict()
            if "S" in inp_init:
                self.initials["SEIR"]["S"] = inp_init.get("S")
            else:
                self.initials["SEIR"]["S"] = 1000        

            if "E" in inp_init:
                self.initials["SEIR"]["E"] = inp_init.get("E")
            else:
                self.initials["SEIR"]["E"] = 0

            if "I" in inp_init:
                self.initials["SEIR"]["I"] = inp_init.get("I")
            else:
                self.initials["SEIR"]["I"] = int(halfcauchy.rvs(loc=3))

            if "R" in inp_init:
                self.initials["SEIR"]["R"] = inp_init.get("R")
            else:
                self.initials["SEIR"]["R"] = 0

            if "epsilon" in inp_init:
                self.initials["SEIR"]["epsilon"] = inp_init.get("epsilon")
            else:
                self.initials["SEIR"]["epsilon"] = np.random.uniform(0, 1)

            if "gamma" in inp_init:
                self.initials["SEIR"]["gamma"] = inp_init.get("gamma")
            else:
                self.initials["SEIR"]["gamma"] = np.random.uniform(0, 1)


        def _generate_change_points():
            if "lambda_0" in inp_init:
                self.initials["lambda_0"] = inp_init.get("lambda_0")
            else:
                self.initials["lambda_0"] = np.random.uniform(0, 1)

            if "change_points" in inp_init:
                self.initials["change_points"] = inp_init.get("change_points")
            else:
                self.initials["change_points"] = None #No change points


        self.initials = dict()
        _generate_compartmental()
        _generate_change_points()

        return self.initials



    def _update_initial(self, **initial_values):
        """
            Generates the initial values for all of the attributes that get used by the following functions.
            If they are not supplied, they are generated by random. Some are random values are drawon from
            a uniform distribution, some from uniform normal distribution and even other
            ones are just set to a fixed value.

            They can be set/changed by editing the attributes before running :py:meth:`generate` or by passing
            a keyword dict into the class at :py:meth:`__init__`.
        """

        def _generate_change_points_we():
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


        self.initials = {}

        if "S_initial" in initial_values:
          self.initials["S_initial"] = initial_values.get("S_initial")
        else:
          self.initials["S_initial"] = 1000

        if "E_initial" in initial_values:
          self.initials["E_initial"] = initial_values.get("E_initial")
        else:
          self.initials["E_initial"] = 0

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
          self.initials["change_points"] = _generate_change_points_we()

        if "noise_factor" in initial_values:
          self.initials["noise_factor"] = initial_values.get("noise_factor")
        else:
          self.initials["noise_factor"] = 0.00001

        if "offset_sunday" in initial_values:
          self.initials["offset_sunday"] = initial_values.get("offset_sunday")
        else:
          d = self.data_begin
          offset = 0
          while d.weekday() != 6: # 0=Monday 6=Sunday
              d += datetime.timedelta(days=1)
              offset += 1
          self.initials["offset_sunday"] = offset

        if "weekend_factor" in initial_values:
          self.initials["weekend_factor"] = initial_values.get("weekend_factor")
        else:
          self.initials["weekend_factor"] = np.random.uniform(0.1, 0.5)


        if "case_delay" in initial_values:
          self.initials["case_delay"] = initial_values.get("case_delay")
        else:
          self.initials["case_delay"] = 6

        #SEIR
        if "alpha" in initial_values:
          self.initials["alpha"] = initial_values.get("alpha")
        else:
          self.initials["alpha"] = 14 #14 days incubation

        if "gamma" in initial_values:
          self.initials["gamma"] = initial_values.get("gamma")
        else:
          self.initials["gamma"] = 0.4          

        return self.initials


    @classmethod
    def get_context(cls):
        if len(ctx)>0:
            return ctx[-1]
        log.Error("Define a model first")

def use_model_ctx(func):
    """
        Should be used as decorator which returns the
        given model or tries to get the last used context if 
        none is supplied.
    """
    def wrapper(*args,**kwargs):
        if "model" not in kwargs:
            kwargs["model"] = DummyModel.get_context()
        return func(*args,**kwargs)
    return wrapper
