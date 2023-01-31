import pandas as pd
import datetime
import logging
import numpy as np

# Import base class
from .. import Retrieval, get_data_dir, _data_dir_fallback

import urllib, json
import requests
import csv

log = logging.getLogger(__name__)


class France(Retrieval):
    """
    This class can be used to retrieve and filter the dataset from the online repository of the `France Government <https://www.data.gouv.fr/>`_.

    `Cases <https://www.data.gouv.fr/fr/datasets/taux-dincidence-de-lepidemie-de-covid-19//>`_.
    Deaths per age group sadly not available anymore.


    Features
        - download the full dataset
        - filter by date
        - filter by deaths and confirmed cases
        - filter by age group


    """

    def __init__(self, auto_download=False):
        """
        On init of this class the base Retrieval Class __init__ is called, with jhu specific
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
        name = "France_data_gouv_fr"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = "https://www.data.gouv.fr/fr/datasets/r/19a91d64-3cd3-42fc-9943-d635491a4d76"

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Surpress warning

        """
        Fallbacks
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
        Attempts to download from the main urls (self.url_csv) which was set on initialization of
        this class.
        If this fails it downloads from the fallbacks. It can also be specified to use the local files
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
        """
        Converts the data to a usable format i.e. converts all date string to
        datetime objects and some other column names.

        This is most of the time the first place one has to look at if something breaks!

        self.data -> self.data converted
        """

        def helper(df):
            try:
                df = df.rename(
                    columns={"jour": "date", "P": "confirmed", "cl_age90": "age_group"}
                )
                df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
                df = df.set_index(["date", "age_group"])
            except Exception as e:
                log.warning(f"There was an error formating the data! {e}")
                raise e
            return df

        self.data = helper(self.data)
        self.data.name = "confirmed"
        self.data["confirmed"] = self.data["confirmed"].astype(int)

        return True

    def get_new(
        self,
        value="confirmed",
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
        age_group: str = None,
    ):
        """
        Retrieves all new cases from the Johns Hopkins University dataset as a DataFrame with datetime index.
        Can be filtered by value, country and state, if only a country is given all available states get summed up.

        Parameters
        ----------
        value: str
            Which data to return, possible values are
            - "confirmed",
            (default: "confirmed")
        begin_date : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used
        end_date : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used
        age_group : optional
            Possible: '09', '19', '29', '39', '49', '59', '69', '79', '89', '90', '0'
        Returns
        -------
        : pandas.DataFrame
            table with new cases and the date as index

        """

        # ------------------------------------------------------------------------------ #
        # Default Parameters
        # ------------------------------------------------------------------------------ #
        if value not in ["confirmed"]:
            raise ValueError('Invalid value. Valid options: "confirmed"')

        if self.data is None:
            self.download_all_available_data()

        # If no date is given set to first and last dates in data
        if data_begin is None:
            data_begin = self.__get_first_date()
        if data_end is None:
            data_end = self.__get_last_date()

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #
        df = pd.DataFrame(columns=[value])

        # Select by age_group
        if age_group is not None:
            df = (
                self.data.xs(age_group, level="age_group")
                .groupby(level="date")[value]
                .sum()
            )
        else:
            df = self.data[value]
            df.index = df.index.droplevel(level="age_group")
            df = df.groupby("date").sum()

        if value == "deaths":
            df = df.drop(df.index[0])

        df = self.filter_date(df, data_begin, data_end)
        return df

    def filter_date(
        self,
        df,
        begin_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
    ):
        """
        Returns give dataframe between begin and end date. Dataframe has to have a datetime index.

        Parameters
        ----------
        begin_date : datetime.datetime, optional
            First day that should be filtered
        end_date : datetime.datetime, optional
            Last day that should be filtered

        Returns
        -------
        : pandas.DataFrame
        """
        if begin_date is None:
            begin_date = self.__get_first_date()
        if end_date is None:
            end_date = self.__get_last_date()

        if not isinstance(begin_date, datetime.datetime) and isinstance(
            end_date, datetime.datetime
        ):
            raise ValueError(
                "Invalid begin_date, end_date: has to be datetime.datetime object"
            )

        return df[begin_date:end_date]

    def __get_first_date(self):
        return self.data.index.get_level_values(level="date").min()

    def __get_last_date(self):
        return self.data.index.get_level_values(level="date").max()

    # ------------------------------------------------------------------------------ #
    # Helper methods, overload from the base class
    # ------------------------------------------------------------------------------ #

    def _download_csv_from_source(self, filepath, **kwargs):

        with requests.Session() as s:
            download = s.get(filepath)

            decoded_content = download.content.decode("utf-8")
            cr = csv.reader(decoded_content.splitlines(), delimiter=";")

        my_list = list(cr)
        df = pd.DataFrame(my_list)
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        self.data = df
