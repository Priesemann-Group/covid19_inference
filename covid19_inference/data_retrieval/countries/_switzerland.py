import pandas as pd
import datetime
import logging
import numpy as np

# Import base class
from .. import Retrieval, get_data_dir, _data_dir_fallback

import urllib, json


log = logging.getLogger(__name__)


class Switzerland(Retrieval):
    """
    This class can be used to retrieve and filter the dataset from Switzerland.
    Age group data is only avaliable for the caton zürich at the moment.
    The data gets retrieved from `github <https://github.com/openZH/covid_19>`_ .

    Features
        - download the full dataset
        - filter by date
        - filter by deaths and confirmed cases
        - filter by age group

    Example
    -------
    .. code-block::
        TODO
    """

    def __init__(self, auto_download=False):
        """
        On init of this class the base Retrieval Class __init__ is called, with rki specific
        arguments.

        Parameters
        ----------
        auto_download : bool, optional
            Whether or not to automatically call the download_all_available_data() method.
            One should explicitly call this method for more configuration options
            (default: false)
        """

        # ------------------------------------------------------------------------------ #
        #  Init Retrieval Base Class
        # ------------------------------------------------------------------------------ #
        """
        A name mainly used for the Local Filename
        """
        name = "Switzerland"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = r"https://raw.githubusercontent.com/openZH/covid_19/master/fallzahlen_kanton_alter_geschlecht_csv/COVID19_Fallzahlen_Kanton_ZH_alter_geschlecht.csv"

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Surpress warning

        """
        fallback array can be anything a filepath or callable methods
        """
        fallbacks = [
            _data_dir_fallback + "/" + name + "_fallback.csv.gz",
        ]
        """
        If the local file is older than the update_interval it gets updated once the
        download all function is called. Can be diffent values depending on the parent class
        """
        update_interval = datetime.timedelta(days=1)

        # Init the retrieval base class
        Retrieval.__init__(self, name, url_csv, fallbacks, update_interval, **kwargs)

        self.data = None

        if auto_download:
            self.download_all_available_data()

    def download_all_available_data(self, force_local=False, force_download=False):
        """
        Attempts to download from the main url (self.url_csv) which was given on initialization.
        If this fails download from the fallbacks. It can also be specified to use the local files
        or to force the download. The download methods get inhereted from the base retrieval class.

        Parameters
        ----------
        force_local : bool, optional
            If True forces to load the local files.
        force_download : bool, optional
            If True forces the download of new files
        """
        if force_local and force_download:
            raise ValueError("force_local and force_download cant both be True!!")

        # ------------------------------------------------------------------------------ #
        # 1 Download or get local file
        # ------------------------------------------------------------------------------ #
        retrieved_local = False
        if self._timestamp_local_old(force_local) or force_download:
            self._download_helper(**self.kwargs)
        else:
            retrieved_local = self._local_helper()

        # ------------------------------------------------------------------------------ #
        # 2 Save local
        # ------------------------------------------------------------------------------ #
        self._save_to_local() if not retrieved_local else None

        # ------------------------------------------------------------------------------ #
        # 3 Convert to useable format
        # ------------------------------------------------------------------------------ #
        self._to_iso()

    def _to_iso(self):
        df = self.data

        df = df.rename(columns={"Date": "date", "confirmados": "cumulative_cases",})
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df = df.set_index("date")
        self.data = df.sort_index()

    def get_new(
        self,
        value="confirmed",
        age_group: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
    ):
        """
        Retrieves all new cases from the Belgian dataset as a DataFrame with datetime index.
        Can be filtered by value, region and province, if only a region is given all available provinces get summed up.

        Parameters
        ----------
        value: str
            Which data to return, possible values are
            - "confirmed",
            - "deaths"
            (default: "confirmed")
        age_group: str
            Which age group to return, possible format: "number-number" inclusive i.e. [num1,num2]
        begin_date : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used
        end_date : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used

        Returns
        -------
        : pandas.DataFrame
            table with new cases and the date as index

        """

        # ------------------------------------------------------------------------------ #
        # Default Parameters
        # ------------------------------------------------------------------------------ #
        assert value in ["confirmed", "deaths",], f"Value '{value}' not possible!"

        if self.data is None:
            self.download_all_available_data()

        if data_begin is None:
            data_begin = self.__get_first_date()
        if data_end is None:
            data_end = self.__get_last_date()

        # ------------------------------------------------------------------------------ #
        # Filter data
        # ------------------------------------------------------------------------------ #

        df = self.data

        # Check for tests since the data is structured another way
        if value == "confirmed":
            column = "NewConfCases"
        if value == "deaths":
            column = "NewDeaths"

        if age_group is None:
            df = df[column].groupby("date").sum()[data_begin:data_end]
            df = pd.DataFrame(df)
            df.columns = [("Switzerland", "all age groups")]
            return df

        # Age group
        num1, num2 = age_group.split("-")
        df = df[(df["AgeYear"] >= int(num1)) & (df["AgeYear"] <= int(num2))]

        df = df[column].groupby("date").sum()[data_begin:data_end]
        df = pd.DataFrame(df)
        df.columns = [("Switzerland", age_group)]
        return df

    def get_total(
        self,
        value="confirmed",
        age_group: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
    ):
        """
        Retrieves all cumulative cases from the Belgian dataset as a DataFrame with datetime index.

        Parameters
        ----------
        value: str
            Which data to return, possible values are
            - "confirmed",
            - "deaths"
            (default: "confirmed")
        age_group: str
            Which age group to return, possible format: "number-number" inclusive i.e. [num1,num2]
        begin_date : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used
        end_date : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used

        Returns
        -------
        : pandas.DataFrame
            table with new cases and the date as index
        """
        return self.get_new(
            value=value, age_group=age_group, data_begin=data_begin, data_end=data_end,
        ).cumsum()

    def __get_first_date(self):
        return self.data.index.min()

    def __get_last_date(self):
        return self.data.index.max()
