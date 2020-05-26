# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-05-26 13:09:09
# @Last Modified: 2020-05-26 15:03:58
# ------------------------------------------------------------------------------ #

from model import *
from scipy.stats import lognorm
import numpy as np

@use_model_ctx
def SIR(dt = 0.1, model = None):

    # SIR model as vector
    def f(t,y,lambda_t):
        return lambda_t * np.array([-1.0, 1.0, 0.0]) * y[0] * y[1] / N + np.array(
            [0, -model.mu * y[1], model.mu * y[1]]
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
    N = (
        model.initials["S_initial"]
        + model.initials["I_initial"]
        + model.initials["R_initial"]
        )

    t_n = [0]

    y_n = [
        np.array(
            [
            model.initials["S_initial"],
            model.initials["I_initial"],
            model.initials["R_initial"],
            ]
        )
        ]

    # ------------------------------------------------------------------------------ #
    # Timesteps
    # ------------------------------------------------------------------------------ #

    t_stop = model.data_len # When to stop the RK integration

    i = 0
    t = 0 # Keep record of the current t value
    while t < t_stop:
        t_n.append(t_n[i] + dt)
        y_n.append(RK4(dt, t_n[i], y_n[i], 0.5))
        t = t + dt
        i = i+1
    return t_n ,y_n



@use_model_ctx
def SEIR(dt = 0.01, model=None, convolve=False):
    def f(t,y,lambda_t):
        """
            Parameters
            ----------
            t : time t
                not needed but here for consistency in RK4
            y : state_vector
                containing S_t,E_t,I_t,R_t
            lambda_t : number
                lambda at time t
        """
        S = y[0]
        E = y[1]
        I = y[2]
        R = y[3]

        dS = model.mu * N - model.mu * S - lambda_t * I / N * S
        dE = lambda_t * I / N * S - (model.mu + model.initials["alpha"]) * E
        dI = model.initials["alpha"] * E - (model.initials["gamma"] + model.mu) * I
        dR = model.initials["gamma"] * I - model.mu * R
        return np.array([dS,dE,dI,dR])


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
    N = (
        model.initials["S_initial"]
        + model.initials["E_initial"]
        + model.initials["I_initial"]
        + model.initials["R_initial"]
    )

    t_n = [0]

    y_timeseries = [ 
        np.array(
            [
                model.initials["S_initial"],
                model.initials["E_initial"],
                model.initials["I_initial"],
                model.initials["R_initial"],
            ]
        )
        ]
    # ------------------------------------------------------------------------------ #
    # Timesteps
    # ------------------------------------------------------------------------------ #

    t_stop = model.data_len # When to stop the RK integration
    
    i = 0 # Index for arrays
    t = 0 # Keep record of the current t value
    while t < t_stop:
        t_n.append(t_n[i] + dt)
        y_timeseries.append(RK4(dt, t_n[i], y_timeseries[i], 0.8))
        t = t + dt
        i = i+1
    return t_n ,y_timeseries




import datetime
#Create dates for data generation
data_begin = datetime.datetime(2020,3,5)
data_end = datetime.datetime(2020,5,3)

dd = DummyModel(data_begin,data_end, mu=0.13)
import matplotlib.pyplot as plt
x,y = SEIR()
plt.plot(x,y)
plt.xlim(0,30)
plt.show()