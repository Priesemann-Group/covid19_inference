import pandas as pd
import numpy as np
import os
from covid19_inference import plotting

from .data_retrieval.retrieval import set_data_dir,get_data_dir
import datetime

import logging
log = logging.getLogger(__name__)


# Low level save trace
def save_trace(
    trace,
    dirname=None,
    varnames=None
    ):
    """
    Saves the given trace into a director inside the global data directory. One can set the global directory
    with `set_data_dir()`, defaults to os dependent temp folder.


    Parameters
    ----------
    trace : pymc3 multitrace
        The give trace which should be saved.
    dirname : str, optional
        Filename for the saved files defaults to trace_backup_%timestamp%
    varnames : arry of strings, optional
        Defaults to all available vars


    Returns
    -------
    :str
        Full filepath to the trace files
    """

    """ Default parameters
    """
    
    if dirname is None:
        dirname = "trace_backup_"+datetime.datetime.now().strftime("%y-%m-%d-%H:%M:%S")

    if varnames is None:
        varnames = trace.varnames

    """ Save data
    """
    # Get full path of directory
    directory = get_data_dir()+dirname

    # Check if dir exists (should never happen) and create it if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Iterate over all wanted vars in the varnames
    for varname in varnames:
        try:
            # Load trace var as dataframe
            df = pd.DataFrame(trace[str(varname)])
        except Exception as e:
            log.warning(f"Varname '{varname} not found in trace! [{e}]")
        try:
            # Save dataframe
            f = open(directory+"/"+str(varname), "w+")
            f.write(df.to_csv(index=False))
            f.close()
        except Exception as e:
            log.warning(f"Varname '{varname}' could not be saved! [{e}]")

    return directory


def load_trace_as_dict(
    dirname =None,
    varnames=None
    ):
    """
    Loads a trace which was earlier saved with the `save_trace` function.

    Parameters
    ----------
    directory : str, optional
        Name of the desired trace e.g. `trace_backup_%timestamp%`, defaults to the last saved one.
        (Only works if the default name was used to save the trace)
    varnames : array of strings, optional
        If one only wants to return specific vars from the saved trace, 
        defaults to all available vars
    """

    """ Default parameters
    """
    return_date = None
    if dirname is None:
        subfolders = [os.path.basename(f.path) for f in os.scandir(get_data_dir()) if f.is_dir() ]
        for folder in subfolders:
            if "trace_backup_" not in folder:
                continue
            # Get date of folder
            str_time = folder[-17:]
            timestamp = datetime.datetime.strptime(str_time, "%y-%m-%d-%H:%M:%S")
            # If the timestamp is newer overwrite the dirname
            if return_date is None or timestamp > return_date:
                return_date = timestamp
                dirname = "trace_backup_"+timestamp.strftime("%y-%m-%d-%H:%M:%S")
        if return_date is None:
            log.error(f"No trace saved in {get_data_dir()}!")

    if varnames is None:
        varnames = [os.path.basename(f.path) for f in os.scandir(get_data_dir()+dirname)]


    trace = dict()
    for varname in varnames:
        trace[varname] = pd.read_csv(get_data_dir()+dirname+"/"+varname)

    return trace



def update_collection(
    country,
    dataset="unknown",
    trace=None,
    varnames=None,
    change_points=None,
    other_vars=None,
):
    """
    Saves the given data into a directory
    
    Parameters
    ----------
    filename: str
    dataset: str
    country: str
    trace: pymc3 multitrace
    varnames: put something that is conversible to str, e.g. model.unobserved_RVs for a pymc3 model
    change_points: dictionary
    other_vars: dictionary, make sure to include the variables that you need later, e.g.
    other_vars = {"bd": bd, "ed": ed, 
              "diff_data_sim": diff_data_sim, "num_days_forecast": num_days_forecast, 
              "lockdown_date": lockdown_date, "lockdown_type": lockdown_type}
    in the usual scripts
    
    Returns
    -------
    None
    """
    try:
        os.chdir("results_collection")
    except FileNotFoundError:
        os.mkdir("results_collection")
        os.chdir("results_collection")

    try:
        os.chdir(country)
    except FileNotFoundError:
        os.mkdir(country)
        os.chdir(country)

    try:
        os.chdir(dataset)
    except FileNotFoundError:
        os.mkdir(dataset)
        os.chdir(dataset)

    for varname in varnames:
        try:
            df = pd.DataFrame(trace[str(varname)])
        except:
            print("variable not found in trace")
        try:
            f = open(str(varname), "w+")
            f.write(df.to_csv(index=False))
            f.close()
        except TypeError:
            print("bad varname format, nothing written")

    try:
        df = pd.DataFrame(other_vars, index=[0])
        f = open("other_vars", "w+")
        f.write(df.to_csv(index=False))
        f.close()
    except TypeError:
        print("other_vars not a dictionary")

    try:
        df = pd.DataFrame(change_points)
        f = open("change_points", "w+")
        f.write(df.to_csv(index=False))
        f.close()
    except TypeError:
        print("change_points not a dictionary or index not found")

    os.chdir("..")
    os.chdir("..")
    os.chdir("..")

    return


def read_variable(country, dataset, variable):
    try:
        os.chdir("results_collection")
    except FileNotFoundError:
        print("Collection not found")
        return

    try:
        os.chdir(country)
    except FileNotFoundError:
        print("Country not found")
        os.chdir("..")
        return

    try:
        os.chdir(dataset)
    except FileNotFoundError:
        print("Dataset not found")
        os.chdir("..")
        os.chdir("..")
        return

    try:
        df = pd.read_csv(variable)

    except FileNotFoundError:
        print("variable not found")
        df = None

    os.chdir("..")
    os.chdir("..")
    os.chdir("..")
    return df


def read_variable_country(country, variable):
    try:
        os.chdir("results_collection")
    except FileNotFoundError:
        print("Collection not found")
        return

    try:
        os.chdir(country)
    except FileNotFoundError:
        print("Country not found")
        os.chdir("..")
        return

    datasets = os.listdir()
    os.chdir("..")
    os.chdir("..")

    dfdict = {}
    for dataset in datasets:
        dfdict[dataset] = read_variable(country, dataset, variable)

    return dfdict


def get_countries():
    try:
        os.chdir("results_collection")
    except FileNotFoundError:
        print("Collection not found")
        return

    countries = os.listdir()
    os.chdir("..")

    return countries
