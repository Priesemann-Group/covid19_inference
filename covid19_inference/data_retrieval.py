import datetime
import os

import numpy as np
import pandas as pd

import urllib, json

_format_date = lambda date_py: "{}/{}/{}".format(
    date_py.month, date_py.day, str(date_py.year)[2:4]
)



def get_last_date(data_df):
    last_date = data_df.columns[-1]
    month, day, year = map(int, last_date.split("/"))
    return datetime.datetime(year + 2000, month, day)

def get_first_date(data_df):
    last_date = data_df.columns[-1]
    month, day, year = map(int, last_date.split("/"))
    return datetime.datetime(year + 2000, month, day)


class Johns_hopkins_university():
    """
    Contains all functions for downloading, filtering and manipulating data from the Johns Hopkins University.
    Automatically downloads all files from the online repository of the Coronavirus Visual Dashboard operated by the Johns Hopkins University.

    Parameters
    ----------
    auto_download : bool, optional
        whether or not to automatically download the data from jhu (default: false)

    """
    def __init__(self, auto_download = False):
        self.confirmed, self.deaths, self.recovered : pd.DataFrame

        if auto_download:
            self.download_all_available_data()


    def download_all_available_data(self,
        fp_confirmed:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
        fp_deaths:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        fp_recovered:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
        save_to_attributes:bool=True)-> (pd.DataFrame,pd.DataFrame,pd.DataFrame):
        """
        Attempts to download the most current data for the confirmed cases, deaths and recovered cases from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        (and falls back to the backup provided with our repo if it fails TODO).
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        fp_confirmed,fp_deaths,fp_recovered : str, optional
            filepath or URL pointing to the original CSV of global confirmed cases, deaths or recovered cases
        save_to_attributes : bool, optional
            Should the returned dataframe tuple be saved as attributes (default:true)

        Returns
        -------
        : pandas.DataFrame tuple
            tuple of table with confirmed, deaths and recovered cases
        """

        return self.download_confirmed(fp_confirmed, save_to_attributes), self.download_deaths(fp_deaths, save_to_attributes), self.download_recovered(fp_recovered, save_to_attributes)

    def download_confirmed(self,
        fp_confirmed:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
        save_to_attribues:bool=True
        ):
        """
        Attempts to download the most current data for the confirmed cases from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        (and falls back to the backup provided with our repo if it fails TODO).
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        fp_confirmed : str, optional
            filepath or URL pointing to the original CSV of global confirmed cases, deaths or recovered cases
        save_to_attributes : bool, optional
            Should the returned dataframe tuple be saved as attributes (default:true)

        Returns
        -------
        : pandas.DataFrame tuple
            tuple of table with confirmed, deaths and recovered cases
        """
        confirmed = self.__to_iso(self.__download_from_source(fp_confirmed))
        if save_to_attribues:
            self.confirmed = confirmed
        return confirmed

    def download_deaths(self,
        fp_deaths:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        save_to_attribues:bool=True
        ):
        """
        Attempts to download the most current data for the deaths from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        (and falls back to the backup provided with our repo if it fails TODO).
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        fp_deaths : str, optional
            filepath or URL pointing to the original CSV of global confirmed cases, deaths or recovered cases
        save_to_attributes : bool, optional
            Should the returned dataframe tuple be saved as attributes (default:true)

        Returns
        -------
        : pandas.DataFrame tuple
            tuple of table with confirmed, deaths and recovered cases
        """
        deaths = self.__to_iso(self.__download_from_source(fp_deaths))
        if save_to_attribues:
            self.deaths = deaths
        return deaths

    def download_recovered(self,
        fp_recovered:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
        save_to_attribues:bool=True
        ):
        """
        Attempts to download the most current data for the recovered cases from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        (and falls back to the backup provided with our repo if it fails TODO).
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        fp_recovered : str, optional
            filepath or URL pointing to the original CSV of global confirmed cases, deaths or recovered cases
        save_to_attributes : bool, optional
            Should the returned dataframe tuple be saved as attributes (default:true)

        Returns
        -------
        : pandas.DataFrame tuple
            tuple of table with confirmed, deaths and recovered cases
        """
        recovered = self.__to_iso(self.__download_from_source(fp_recovered))
        if save_to_attribues:
            self.recovered = recovered
        return recovered

    def __download_from_source(self, url, fallback = None)-> pd.DataFrame:
        """
        Private method
        Downloads one csv file from an url and converts it into a pandas dataframe. A fallback source can also be given.

        Parameters
        ----------
        url : str
            Where to download the csv file from
        fallback : str, optional
            Fallback source for the csv file, filename of file that is located in /data/

        Returns
        -------
        : pandas.DataFrame
            Raw data from the source url as dataframe
        """
        try:
            data = pd.read_csv(url, sep=",")
        except Exception as e:
            print("Failed to download current data 'confirmed cases', using local copy.")
            this_dir = os.path.dirname(__file__)
            data = pd.read_csv(
                this_dir + "/../data/" + fallback, sep=","
            )
        return data

    def __to_iso(self, df) -> pd.DataFrame:
        """
        Convert Johns Hopkins University dataset to nicely formatted DataFrame.
        Drops Lat/Long columns and reformats to a multi-index of (country, state).

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe to convert to the iso format

        Returns
        -------
        : pandas.DataFrame
        """

        # change columns & index
        df = df.drop(columns=['Lat', 'Long']).rename(columns={
            'Province/State': 'state',
            'Country/Region': 'country'
        })
        df = df.set_index(['country', 'state'])
        # datetime columns
        df.columns = [datetime.datetime.strptime(d, '%m/%d/%y') for d in df.columns]
        return df


    def get_confirmed_deaths_recovered(self, country:str = None, state:str = None, begin_date:str = None, end_date:str = None) -> pd.DataFrame:
        """
        Retrieves all confirmed, deaths and recovered Johns Hopkins University dataset as a DataFrame with datetime index.
        Can be filtered by country and state, if only a country is given all available states get summed up.

        Parameters
        ----------
        country : str, optional
            name of the country (the "Country/Region" column), can be None if state is set
        state : str, optional
            name of the state (the "Province/State" column), can be None if country is set
        begin_date : str, optional
            First day that should be filtered, in format '%m/%d/%y' 
        end_date : str, optional
            Last day that should be filtered, in format '%m/%d/%y' 

        Returns
        -------
        : pandas.DataFrame
        """

        # filter
        df = pd.DataFrame(columns=['date', 'confirmed', 'deaths', 'recovered']).set_index('date')
        if country is None:
            df['confirmed'] = self.confirmed.sum()
            df['deaths'] = self.deaths.sum()
            df['recovered'] = self.recovered.sum()       
        else:
            if state is None:
                df['confirmed'] = self.confirmed.loc[country].sum()
                df['deaths'] = self.deaths.loc[country].sum()
                df['recovered'] = self.recovered.loc[country].sum()      
            else:
                df['confirmed'] = self.confirmed.loc[(country,state)]
                df['deaths'] = self.deaths.loc[(country,state)]
                df['recovered'] = self.recovered.loc[(country,state)]   

        df.index.name = 'date'

        df = self.filter_date(df, begin_date, end_date)

        return df

    def get_confirmed(self, country:str = None, state:str = None, begin_date:str = None, end_date:str = None) -> pd.DataFrame:
        """
        Attempts to download the most current data for the confirmed cases from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        and falls back to the backup provided with our repo if it fails.
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        country : str, optional
            name of the country (the "Country/Region" column), can be None
        state : str, optional
            name of the state (the "Province/State" column), can be None
        begin_date : str, optional
            First day that should be filtered, in format '%m/%d/%y' 
        end_date : str, optional
            Last day that should be filtered, in format '%m/%d/%y' 

        Returns
        -------
        : pandas.DataFrame
            table with confirmed cases and the date as index
        """

        if country == "None":
            country = None
        if state == "None":
            state = None

        df = pd.DataFrame(columns=['date', 'confirmed']).set_index('date')
        if country is None:
            df['confirmed'] = self.confirmed.sum()
        else:
            if state is None:
                df['confirmed'] = self.confirmed.loc[country].sum()              
            else:
                df['confirmed'] = self.confirmed.loc[(country,state)]
        df.index.name = 'date'

        return df

    def get_deaths(self, country:str = None, state:str = None, begin_date:str = None, end_date:str = None) -> pd.DataFrame:
        """
        Attempts to download the most current data for the confirmed deaths from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        and falls back to the backup provided with our repo if it fails.
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        country : str, optional
            name of the country (the "Country/Region" column), can be None
        state : str, optional
            name of the state (the "Province/State" column), can be None
        begin_date : str, optional
            First day that should be filtered, in format '%m/%d/%y' 
        end_date : str, optional
            Last day that should be filtered, in format '%m/%d/%y' 

        Returns
        -------
        : pandas.DataFrame
            table with confirmed cases and the date as index
        """
        if country == "None":
            country = None
        if state == "None":
            state = None

        df = pd.DataFrame(columns=['date', 'deaths']).set_index('date')
        if country is None:
            df['deaths'] = self.deaths.sum()
        else:
            if state is None:
                df['deaths'] = self.deaths.loc[country].sum()              
            else:
                df['deaths'] = self.deaths.loc[(country,state)]
        
        df.index.name = 'date'

        return df

    def get_recovered(self, country:str = None, state:str = None, begin_date:str = None, end_date:str = None) -> pd.DataFrame:
        """
        Attempts to download the most current data for the confirmed recoveries from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        and falls back to the backup provided with our repo if it fails.
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        country : str, optional
            name of the country (the "Country/Region" column), can be None 
        state : str, optional
            name of the state (the "Province/State" column), can be None
        begin_date : str, optional
            First day that should be filtered, in format '%m/%d/%y' 
        end_date : str, optional
            Last day that should be filtered, in format '%m/%d/%y' 

        Returns
        -------
        : pandas.DataFrame
            table with recovered cases and the date as index
        """
        if country == "None":
            country = None
        if state == "None":
            state = None

        df = pd.DataFrame(columns=['date', 'recovered']).set_index('date')
        if country is None:
            df['recovered'] = self.recovered.sum()
        else:
            if state is None:
                df['recovered'] = self.recovered.loc[country].sum()              
            else:
                df['recovered'] = self.recovered.loc[(country,state)]

        df.index.name = 'date'

        return df

    def filter_date(self, df,  begin_date:str = None, end_date:str=None) -> pd.DataFrame:
        """
        Returns give dataframe between begin and end date. Dataframe has to have a datetime index.

        Parameters
        ----------
        begin_date : str, optional
            First day that should be filtered, in format '%m/%d/%y' 
        end_date : str, optional
            Last day that should be filtered, in format '%m/%d/%y' 

        Returns
        -------
        : pandas.DataFrame
        """
        if begin_date is None:
            begin_date = self.__get_first_date(df)
        else:
            begin_date = datetime.datetime.strptime(begin_date, '%m/%d/%y')
        if end_date is None:
            end_date = self.__get_last_date(df)
        else:
            end_date = datetime.datetime.strptime(end_date, '%m/%d/%y')
        print(begin_date)
        print(df[begin_date:end_date])


    def __get_first_date(self,df):
        return df.index[0]
    def __get_last_date(self,df):
        return df.index[-1]