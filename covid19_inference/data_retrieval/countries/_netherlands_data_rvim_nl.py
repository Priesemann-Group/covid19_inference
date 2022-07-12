import pandas as pd
import datetime
import logging
import numpy as np

# Import base class
from .. import Retrieval, get_data_dir, _data_dir_fallback

import urllib, json


log = logging.getLogger(__name__)


class Netherlands(Retrieval):
    """
    This class can be used to retrieve and filter the dataset from the offical netherlands data source.
    The data gets retrieved from `data.rivm.nl <https://data.rivm.nl/geonetwork/srv/dut/catalog.search#/metadata/2c4357c8-76e4-4662-9574-1deb8a73f724?tab=general>`_ .

    Features
        - download the full dataset
        - filter by date
        - filter by deaths and confirmed cases
        - filter by age group

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
        name = "Netherlands_rvim"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = r"https://data.rivm.nl/covid-19/COVID-19_casus_landelijk.csv"

        """
        Kwargs for pandas read csv
        """
        kwargs = {"sep": ";"}  # Surpress warning

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

        df = df.rename(
            columns={
                "Date_file": "date_sub",
                "Date_statistics": "date",
            }
        )
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S")
        df["date_sub"] = pd.to_datetime(df["date_sub"], format="%Y-%m-%d")
        df = df.set_index("date")
        self.data = df.sort_index()

    def get_new(
        self,
        value="confirmed",
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
        age_group: str = None,
    ):
        """
        Retrieves all cumulative cases from the Robert Koch Institute dataset as a DataFrame with datetime index.
        Can be filtered by value, bundesland and landkreis, if only a country is given all available states get summed up.

        Parameters
        ----------
        value: str
            Which data to return, possible values are
            - "confirmed",
            - "deaths",
            - "hospitalized"
            (default: "confirmed")
        data_begin : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used,
            if none is given could yield errors
        data_end : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used
        age_group : str, optional
            Possible are '0-9','10-19','20-29','30-39','40-49', '50-59','60-69','70-79', '80-89','90+','<50', 'Unknown'

        Returns
        -------
        : pandas.DataFrame
            table with daily new confirmed and the date as index
        """

        # ------------------------------------------------------------------------------ #
        # Default parameters
        # ------------------------------------------------------------------------------ #

        assert value in ["confirmed", "deaths"], f"Value '{value}' not possible!"

        if self.data is None:
            self.download_all_available_data()

        if data_begin is None:
            data_begin = self.__get_first_date()
        if data_end is None:
            data_end = self.__get_last_date()
        header = value
        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #
        if age_group is not None:
            df = self.data.loc[self.data["Agegroup"] == age_group]
            header = header + " " + age_group
        else:
            df = self.data

        if value == "deaths":
            df = df.loc[df["Deceased"] == "Yes"]
        if value == "hospitalized":
            df = df.loc[df["Hospital_admission"] == "Yes"]

        ret = pd.DataFrame()
        ret[header] = df.groupby("date")["Sex"].count()

        return ret[data_begin:data_end]

    def __get_first_date(self):
        return self.data.index.min()

    def __get_last_date(self):
        return self.data.index.max()
