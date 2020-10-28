import pandas as pd
import datetime
import logging
import numpy as np

# Import base class
from .. import Retrieval, get_data_dir

import urllib, json


log = logging.getLogger(__name__)


class Belgium(Retrieval):
    """
    This class can be used to retrieve and filter the dataset from the `Belgian EPISTAT Website <https://epistat.wiv-isp.be/covid/>`_.

    Features
        - download the full dataset
        - filter by date
        - filter by bundesland
        - filter by recovered, deaths and confirmed cases

    Example
    -------
    .. code-block::

        ewi = cov19.data_retrieval.Epistat_wiv_isp()
        ewi.download_all_available_data()

        #Acess the data by
        ewi.data
        #or
        ewi.get_new("confirmed","Sachsen")
        ewi.get_total(filter)
    """

    @property
    def data(self):
        if (
            self.confirmed is None
            or self.hospitalized is None
            or self.deaths is None
            or self.tests is None
        ):
            return None
        return (self.confirmed, self.hospitalized, self.deaths, self.tests)

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
        name = "Belgium_epistat"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = [
            "https://epistat.sciensano.be/Data/COVID19BE_CASES_AGESEX.csv",  # Confirmed
            "https://epistat.sciensano.be/Data/COVID19BE_HOSP.csv",  # Hospitilization
            "https://epistat.sciensano.be/Data/COVID19BE_MORT.csv",  # Deaths
            "https://epistat.sciensano.be/Data/COVID19BE_tests.csv",  # Tests
        ]

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Surpress warning

        """
        fallback array can be anything a filepath or callable methods
        """
        fallbacks = [
            self._fallback_local_backup,
        ]
        """
        If the local file is older than the update_interval it gets updated once the
        download all function is called. Can be diffent values depending on the parent class
        """
        update_interval = datetime.timedelta(days=1)

        # Init the retrieval base class
        Retrieval.__init__(self, name, url_csv, fallbacks, update_interval, **kwargs)

        self.confirmed = None
        self.hospitalized = None
        self.deaths = None
        self.tests = None

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

    def get_new(
        self,
        value="confirmed",
        age_group: str = None,
        province: str = None,
        region: str = None,
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
            - "hospitalized",
            - "deaths"
            - "tests"
            (default: "confirmed")
        age_group: str
            Which age group to return, possible values are '40-49', '60-69', '50-59', '70-79', '10-19', '30-39', '20-29',
            '0-9', nan, '80-89', '90+'
        region : str, optional
            name of the region in Belgium (the "Region" column), can be None
        province : str, optional
            name of the province (the "Province" column), can be None
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
        assert value in [
            "confirmed",
            "hospitalized",
            "deaths",
            "tests",
        ], f"Value '{value}' not possible!"

        if self.data is None:
            self.download_all_available_data()

        if data_begin is None:
            data_begin = self.__get_first_date()
        if data_end is None:
            data_end = self.__get_last_date()

        # ------------------------------------------------------------------------------ #
        # Filter data
        # ------------------------------------------------------------------------------ #

        df = getattr(self, value)

        # Select regions
        if region is not None and "region" in df.columns:
            df = df.loc[df["region"] == region]
        if province is not None and "province" in df.columns:
            df = df.loc[df["province"] == province]

        if value == "tests":
            df = df.drop(columns="TESTS_ALL_POS")
            df = df.groupby("date").sum()
            return df["TESTS_ALL"].asfreq("D", fill_value=0)[data_begin:data_end]

        # Select age group
        if age_group is not None and "age_group" in df.columns:
            df = df.loc[df["age_group"] == age_group]

        df = df.groupby("date").sum()
        df.columns = [("Belgium", age_group)]
        return df[("Belgium", age_group)].asfreq("D", fill_value=0)[data_begin:data_end]

    def get_total(
        self,
        value="confirmed",
        age_group: str = None,
        province: str = None,
        region: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
    ):
        """
        Retrieves all cumulative cases from the Belgian dataset as a DataFrame with datetime index.
        """
        return self.get_new(
            value=value,
            age_group=age_group,
            province=province,
            region=region,
            data_begin=data_begin,
            data_end=data_end,
        ).cumsum()

    def __get_first_date(self):
        return self.confirmed.index.min()

    def __get_last_date(self):
        return self.confirmed.index.max()

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
            self._download_csvs_from_source(
                [
                    get_data_dir() + self.name + "_confirmed" + ".csv.gz",
                    get_data_dir() + self.name + "_hospitilization" + ".csv.gz",
                    get_data_dir() + self.name + "_deaths" + ".csv.gz",
                    get_data_dir() + self.name + "_tests" + ".csv.gz",
                ],
                **self.kwargs,
            )
            log.info(f"Successfully loaded data from local")
            return True
        except Exception as e:
            log.info(f"Failed to load local files! {e} Trying fallbacks!")
            self.download_helper(**self.kwargs)
        return False

    def _save_to_local(self):
        """
        Overloads the method method from the Base Retrival class
        """
        filepaths = [
            get_data_dir() + self.name + "_confirmed" + ".csv.gz",
            get_data_dir() + self.name + "_hospitilization" + ".csv.gz",
            get_data_dir() + self.name + "_deaths" + ".csv.gz",
            get_data_dir() + self.name + "_tests" + ".csv.gz",
        ]
        try:
            self.confirmed.to_csv(filepaths[0], compression="infer", index=False)
            self.hospitalized.to_csv(filepaths[1], compression="infer", index=False)
            self.deaths.to_csv(filepaths[2], compression="infer", index=False)
            self.tests.to_csv(filepaths[3], compression="infer", index=False)
            self._create_timestamp()
            log.info(f"Local backup to {filepaths} successful.")
            return True
        except Exception as e:
            log.warning(f"Could not create local backup {e}")
            raise e
        return False

    def _download_csvs_from_source(self, filepaths, **kwargs):
        self.confirmed = pd.read_csv(filepaths[0], **kwargs)
        self.hospitalized = pd.read_csv(filepaths[1], **kwargs)
        self.deaths = pd.read_csv(filepaths[2], **kwargs)
        self.tests = pd.read_csv(filepaths[3], **kwargs)

    def _fallback_local_backup(self):
        path_confirmed = (
            _data_dir_fallback
            + "/"
            + self.name
            + "_confirmed"
            + "_fallback"
            + ".csv.gz"
        )
        path_hospitilized = (
            _data_dir_fallback
            + "/"
            + self.name
            + "_hospitilization"
            + "_fallback"
            + ".csv.gz"
        )

        path_deaths = (
            _data_dir_fallback + "/" + self.name + "_deaths" + "_fallback" + ".csv.gz"
        )

        path_tests = (
            _data_dir_fallback + "/" + self.name + "_tests" + "_fallback" + ".csv.gz"
        )
        self.confirmed = pd.read_csv(path_confirmed, **self.kwargs)
        self.hospitalized = pd.read_csv(path_hospitilized, **self.kwargs)
        self.deaths = pd.read_csv(path_deaths, **self.kwargs)
        self.tests = pd.read_csv(path_tests, **self.kwargs)

    def _to_iso(self):
        """
        Converts the data to a usable format i.e. converts all date string to
        datetime objects and some other column names.

        This is most of the time the first place one has to look at if something breaks!

        self.data -> self.data converted
        """

        def helper(df):
            try:
                df = df.rename(columns={"DATE": "date",})
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

                if "PROVINCE" in df.columns:
                    df = df.rename(columns={"PROVINCE": "province"})
                if "REGION" in df.columns:
                    df = df.rename(columns={"REGION": "region"})
                if "AGEGROUP" in df.columns:
                    df = df.rename(columns={"AGEGROUP": "age_group"})
                if "CASES" in df.columns:
                    df = df.rename(columns={"CASES": "cases"})
                if "DEATHS" in df.columns:
                    df = df.rename(columns={"DEATHS": "deaths"})
                if "TESTS" in df.columns:
                    df = df.rename(columns={"TESTS": "tests"})
            except Exception as e:
                log.warning(f"There was an error formating the data! {e}")
                raise e
            return df.sort_index()

        self.confirmed = helper(self.confirmed)
        self.hospitalized = helper(self.hospitalized)
        self.deaths = helper(self.deaths)
        self.tests = helper(self.tests)
