# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-06-04 13:20:46
# @Last Modified: 2020-06-17 10:32:50
# ------------------------------------------------------------------------------ #


import datetime
import time as time_module
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import theano
import theano.tensor as tt
import pymc3 as pm

# Now to the fun stuff, we import our module!
try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append("../")
    import covid19_inference as cov19

from covid19_inference.dummydata.change_points import ChangePoint

"""
    ## Change points and model params
"""
lambda_0 = 1.5

mild_dist = ChangePoint(
    lambda_before=lambda_0,
    lambda_after=0.6,
    date_begin=datetime.datetime(2020, 3, 9),
    length=datetime.timedelta(days=2))

strong_dist = ChangePoint(
    lambda_before=0.6,
    lambda_after=0.4,
    date_begin=datetime.datetime(2020, 3, 16),
    length=datetime.timedelta(days=2),
    modulation="step")

contact_ban = ChangePoint(
    lambda_before=0.4,
    lambda_after=0.2,
    date_begin=datetime.datetime(2020, 3, 23),
    length=datetime.timedelta(days=6))

change_points = [mild_dist,strong_dist,contact_ban]

params = dict(
    data_begin = datetime.datetime(2020,3,2),
    data_end = datetime.datetime(2020,4,8),
    mu = 0.13,
    dt = 0.01, #Precicion of the timesteps in RK4 in days
    seed = 101010,
    I=100,
    change_points=change_points)

""" ## Define and run model
"""
model = cov19.dummydata.DummyModel(**params)

#Create lambda_t with sigmoids automaticly gets model
λ_t = cov19.dummydata.generate_lambda_t_from_model()

#Create SIR from lambda_t
t, S, E, I, R = cov19.dummydata.SEIR(λ_t,epsilon=1)

""" ## Plotting
"""

#Create date_time array from t array
# --> Todo make this a build in function
milliseconds = (model.data_end-model.data_begin) / datetime.timedelta(milliseconds=1)
step_size = milliseconds/len(t)
x = pd.date_range(model.data_begin,model.data_end,freq=f"{np.ceil(step_size)}ms")
x = x[:-1]


# Plot it

fig, axes = plt.subplots(2,1)

cov19.plot._timeseries(x,λ_t,ax=axes[0],what="model",color="darkblue")
cov19.plot._format_date_xticks(axes[0])
axes[0].set_ylabel(r"$\lambda_t$")
#cov19.plot._timeseries(x,S,ax=axes[1],what="model",color="tab:red",label="S")
cov19.plot._timeseries(x,E,ax=axes[1],what="model",color="tab:green",label="E")
cov19.plot._timeseries(x,I,ax=axes[1],what="model",color="tab:blue",label="I")
cov19.plot._timeseries(x,R,ax=axes[1],what="model",color="tab:orange",label="R")
cov19.plot._format_date_xticks(axes[1])
axes[1].legend()
axes[1].set_ylabel(r"Cases")