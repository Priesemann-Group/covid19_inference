# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-06-04 12:45:52
# @Last Modified: 2020-06-04 19:15:35
# ------------------------------------------------------------------------------ #
from .model import *
import numpy as np

import logging
log = logging.getLogger(__name__)


@use_model_ctx
def generate_lambda_t_from_model(model=None):
    """
    Creates a lambda_t array which can be used by the model
    """

    log.info("λ_t with sigmoids")
    # ------------------------------------------------------------------------------ #
    # Preliminary parameters
    # ------------------------------------------------------------------------------ #

    #First get cp from model and sort them by date
    # cp = [date, value]
    change_points = model.initials["change_points"]
    #change_points.sort(key=lambda x: x.date_begin)

    # ------------------------------------------------------------------------------ #
    # Calculate lambda array
    # ------------------------------------------------------------------------------ #    

    # Check if first cp before first date
    if change_points[0].date_begin <= model.data_begin:
        raise ValueError(f"Change point cant be before model data begin {change_points[0].date_begin}")

    """
    Go threw every change point and calculate the lambda values between it and
    the following change point
    """

    #Before first change point
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

    #Do last change point
    λ = np.append(λ, change_points[-1].get_lambdas(model.dt)) 

    #Last change point to end of model
    delta = (model.data_end-change_points[-1].date_end).days
    t_cp_to_next_cp = np.arange(0, delta, model.dt)
    λ = np.append(λ,[change_points[-1].lambda_after]*len(t_cp_to_next_cp)) 
    return λ