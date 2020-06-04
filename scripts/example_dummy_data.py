# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-06-04 13:20:46
# @Last Modified: 2020-06-04 14:30:53
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


"""
    Model Params
"""
date_mild_dist_begin = datetime.datetime(2020, 3, 9)
date_strong_dist_begin = datetime.datetime(2020, 3, 16)
date_contact_ban_begin = datetime.datetime(2020, 3, 23)

change_points = [
    [date_mild_dist_begin,0.6],
    [date_strong_dist_begin,0.4],
    [date_contact_ban_begin,0.2]]


params = dict(
    data_begin = datetime.datetime(2020,3,2),
    data_end = datetime.datetime(2020,4,8),
    mu = 0.13,
    dt = 0.1,
    seed = 101010,
    lambda_0 = 0.6,
    change_points=change_points)


"""
    Define model
"""
model = cov19.dummydata.DummyModel(**params)

#Create lambda_t with sigmoids
λ_t = cov19.dummydata.lambda_t_with_sigmoids()

#Create SIR from lambda_t
t, S, I, R = cov19.dummydata.SIR(λ_t)

"""
    Plotting
"""

#Create date_time array from t array
# --> Todo make this a build in function
milliseconds = (model.data_end-model.data_begin) / datetime.timedelta(milliseconds=1)
step_size = milliseconds/len(t)
x = pd.date_range(model.data_begin,model.data_end,freq=f"{np.ceil(step_size)}ms")
x = x[:-1]



# Plot it

fig, axes = plt.subplots(2,1)

cov19.plot._timeseries(x,λ_t,ax=axes[0],what="model")
cov19.plot._format_date_xticks(axes[0])

cov19.plot._timeseries(x,S,ax=axes[1],what="model")
cov19.plot._timeseries(x,I,ax=axes[1],what="model")
cov19.plot._timeseries(x,R,ax=axes[1],what="model")
cov19.plot._format_date_xticks(axes[1])
