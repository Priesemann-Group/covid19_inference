# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-06-04 12:45:52
# @Last Modified: 2020-06-04 14:02:03
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
    change_points.sort(key=lambda x: x[0])
    #Create numpy array for slicing
    change_points = np.array(change_points) 

    start = [model.data_begin,model.initials["lambda_0"]] #Cp at time zero
    end = [model.data_end,change_points[-1][1]]

    # ------------------------------------------------------------------------------ #
    # Calc lambda array
    # ------------------------------------------------------------------------------ #    
    #1. Create t array between changepoint
    #2. calc lambda with that t array and the two changepoint

    # Check if first cp before first date
    if change_points[0][0] <= model.data_begin:
        raise ValueError(f"Change point cant be before model data begin {change_points[0][0]}")

    """Go threw every changepoint and calculate the lambda values between it and
    the following change point"""

    λ = []
    #Before first cp
    λ = np.append(λ,_lambda_between(start,change_points[0]))

    #Between all cps
    for i in range(len(change_points)-1):
        λ = np.append(λ, _lambda_between(change_points[i],change_points[i+1]))

    #Last cp to end of time
    λ = np.append(λ, _lambda_between(change_points[-1],end))

    return λ