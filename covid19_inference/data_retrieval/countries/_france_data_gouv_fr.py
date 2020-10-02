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

    `Deaths <https://www.data.gouv.fr/fr/datasets/donnees-de-certification-electronique-des-deces-associes-au-covid-19-cepidc/>`_.
    
    """

    @property
    def data(self):
        if self.confirmed is None or self.deaths is None:
            return None
        return (
            self.confirmed,
            self.deaths,
        )

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
        url_csv = [
            "https://www.data.gouv.fr/fr/datasets/r/19a91d64-3cd3-42fc-9943-d635491a4d76",
            "https://www.data.gouv.fr/fr/datasets/r/b09da629-e48a-4baa-8d43-9ae28b3248da",
        ]

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Surpress warning

        """
        Fallbacks
        """
        fallbacks = [self._fallback_local_backup]

        """
        If the local file is older than the update_interval it gets updated once the
        download all function is called. Can be diffent values depending on the parent class
        """
        update_interval = datetime.timedelta(days=1)

        # Init the retrieval base class
        Retrieval.__init__(self, name, url_csv, fallbacks, update_interval, **kwargs)

        self.confirmed = None
        self.deaths = None

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
                print(df.columns)
                df = df.set_index(["date", "age_group"])
            except Exception as e:
                log.warning(f"There was an error formating the data! {e}")
                raise e
            return df

        self.confirmed = helper(self.confirmed)
        self.confirmed.name = "confirmed"
        self.confirmed["confirmed"] = self.confirmed["confirmed"].astype(int)

        self.deaths = helper(self.deaths)
        self.deaths["deaths"] = self.deaths["Dc_Elec_Covid_cum"].astype(int).diff()
        self.deaths.name = "deaths"
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
            - "recovered",
            - "deaths"
            (default: "confirmed")
        country : str, optional
            name of the country (the "Country/Region" column), can be None
        state : str, optional
            name of the state (the "Province/State" column), can be None
        begin_date : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used
        end_date : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used
        age_group : optional

        Returns
        -------
        : pandas.DataFrame
            table with new cases and the date as index

        """

        # ------------------------------------------------------------------------------ #
        # Default Parameters
        # ------------------------------------------------------------------------------ #
        if value not in ["confirmed", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        if self.data is None:
            self.download_all_available_data()

        # If no date is given set to first and last dates in data
        if data_begin is None:
            if value == "deaths":
                delta = datetime.timedelta(days=1)
                data_begin = self.__get_first_date() - delta
            else:
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
                getattr(self, value)
                .xs(age_group, level="age_group")
                .groupby(level="date")[value]
                .sum()
            )
        else:
            df = getattr(self, value)[value]
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
        return self.data[0].index.get_level_values(level="date").min()

    def __get_last_date(self):
        return self.data[0].index.get_level_values(level="date").max()

    def get_possible_countries_states(self):
        """
        Can be used to get a list with all possible states and coutries.

        Returns
        -------
        : pandas.DataFrame in the format
        """
        all_entrys = (
            list(self.confirmed.columns)
            + list(self.deaths.columns)
            + list(self.recovered.columns)
        )
        df = pd.DataFrame(all_entrys, columns=["country", "state"])

        return df

    # ------------------------------------------------------------------------------ #
    # Helper methods, overload from the base class
    # ------------------------------------------------------------------------------ #
    def _download_helper(self, **kwargs):
        """
        Overloads the method method from the Base Retrival class
        """
        try:
            # Try to download from original souce
            self._download_csvs_from_source(self.url_csv, **kwargs)
        except Exception as e:
            # Try all fallbacks
            log.info(f"Failed to download from url {self.url_csv} : {e}")
            self._fallback_handler()
        finally:
            # We save it to the local files
            # self.data._save_to_local()
            log.info(f"Successfully downloaded new files.")

    def _local_helper(self):
        """
        Overloads the method method from the Base Retrival class
        """
        try:
            self._download_local_from_source(
                [
                    get_data_dir() + self.name + "_confirmed" + ".csv.gz",
                    get_data_dir() + self.name + "_deaths" + ".csv.gz",
                ],
                **self.kwargs,
            )
            log.info(f"Successfully loaded data from local")
            return True
        except Exception as e:
            log.info(f"Failed to load local files! {e} Trying fallbacks!")
            self._download_helper(**self.kwargs)
        return False

    def _save_to_local(self):
        """
        Overloads the method method from the Base Retrival class
        """
        filepaths = [
            get_data_dir() + self.name + "_confirmed" + ".csv.gz",
            get_data_dir() + self.name + "_deaths" + ".csv.gz",
        ]
        try:
            self.confirmed.to_csv(filepaths[0], compression="infer", index=False)
            self.deaths.to_csv(filepaths[1], compression="infer", index=False)
            self._create_timestamp()
            log.info(f"Local backup to {filepaths} successful.")
            return True
        except Exception as e:
            log.warning(f"Could not create local backup {e}")
            raise e
        return False

    def _download_csvs_from_source(self, filepaths, **kwargs):

        with requests.Session() as s:
            download = s.get(filepaths[0])

            decoded_content = download.content.decode("utf-8")
            cr = csv.reader(decoded_content.splitlines(), delimiter=";")

        my_list = list(cr)
        df = pd.DataFrame(my_list)
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        self.confirmed = df
        with requests.Session() as s:
            download = s.get(filepaths[1])

            decoded_content = download.content.decode("utf-8")
            cr = csv.reader(decoded_content.splitlines(), delimiter=";")

        my_list = list(cr)

        df = pd.DataFrame(my_list)
        df.columns = df.iloc[0]
        df = df.drop(df.index[0])
        self.deaths = df

    def _download_local_from_source(self, filepaths, **kwargs):
        self.confirmed = pd.read_csv(filepaths[0], **kwargs)
        self.deaths = pd.read_csv(filepaths[1], **kwargs)

    def _fallback_local_backup(self):
        path_confirmed = (
            _data_dir_fallback
            + "/"
            + self.name
            + "_confirmed"
            + "_fallback"
            + ".csv.gz"
        )
        path_deaths = (
            _data_dir_fallback + "/" + self.name + "_deaths" + "_fallback" + ".csv.gz"
        )

        self.confirmed = pd.read_csv(path_confirmed, **self.kwargs)
        self.deaths = pd.read_csv(path_deaths, **self.kwargs)
