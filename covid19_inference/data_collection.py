import pandas as pd
import numpy as np
import os
from covid19_inference import plotting


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
