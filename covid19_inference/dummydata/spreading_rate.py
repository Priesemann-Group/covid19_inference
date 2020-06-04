# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-06-04 12:45:52
# @Last Modified: 2020-06-04 18:01:44
# ------------------------------------------------------------------------------ #
from .model import *
import numpy as np

import logging
log = logging.getLogger(__name__)


@use_model_ctx
def lambda_t_with_sigmoids(model=None):
    """
    Creates a lambda_t array which can be used by the model
    """

    def logistics_from_cps(x, cp1_x, cp1_y, cp2_x, cp2_y):
        """
            Calculates the lambda value at the point x
            between cp1_x and cp2_x.

            TODO
            ----
            implement k value 
        """
        L = cp2_y-cp1_y
        C = cp1_y
        x_0 = np.abs(cp1_x - cp2_x)/2 + cp1_x
        #log.debug(f"{L} {C} {x_0} {x}")
        #print(L/(1+np.exp(-4*(x-x_0)))+C)
        return L/(1+np.exp(-2*(x-x_0)))+C

    def _lambda_between(cp1,cp2):
        """
        Helper function to get lambda values between two change points
        """
        #Normalize cp1 to start at x=0
        delta_days = np.abs((cp1[0]-cp2[0]).days)

        #For the timerange between cp1 and cp2 construct each lambda value 
        λ_t_temp = []
        t_temp = np.arange(0,delta_days,model.dt)
        for t in t_temp:
            λ_t_temp.append(logistics_from_cps(t,0,cp1[1],delta_days,cp2[1]))
        return λ_t_temp

    log.info("λ_t with sigmoids")
    # ------------------------------------------------------------------------------ #
    # Preliminar parameters
    # ------------------------------------------------------------------------------ #

    #First get cp from model and sort them by date
    # cp = [date, value]
    change_points = model.initials["change_points"]
    #change_points.sort(key=lambda x: x.date_begin)

    # ------------------------------------------------------------------------------ #
    # Calc lambda array
    # ------------------------------------------------------------------------------ #    

    # Check if first cp before first date
    if change_points[0].date_begin <= model.data_begin:
        raise ValueError(f"Change point cant be before model data begin {change_points[0].date_begin}")

    """Go threw every changepoint and calculate the lambda values between it and
    the following change point"""

    #Before first cp
    delta = (change_points[0].date_begin-model.data_begin).days
    t_begin_to_first_cp = np.arange(0, delta, model.dt)
    λ = [change_points[0].lambda_before]*len(t_begin_to_first_cp)

    for i in range(len(change_points)-1):
        # Gets lambda for the change point
        λ = np.append(λ, change_points[i].get_lambdas(model.dt)) 

        # lambda after change point to begin next change point
        delta = (change_points[i+1].date_begin-change_points[i].date_end).days
        t_cp_to_next_cp = np.arange(0, delta, model.dt)
        λ = np.append(λ,[change_points[i].lambda_after]*len(t_cp_to_next_cp))

    #Do last cp
    λ = np.append(λ, change_points[-1].get_lambdas(model.dt)) 

    #Last cp to end of model time
    delta = (model.data_end-change_points[-1].date_end).days
    t_cp_to_next_cp = np.arange(0, delta, model.dt)
    λ = np.append(λ,[change_points[-1].lambda_after]*len(t_cp_to_next_cp)) 
    return λ