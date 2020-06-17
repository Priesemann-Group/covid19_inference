import datetime
import pandas as pd
import logging

# Import base class
from .retrieval import Retrieval, _data_dir_fallback

log = logging.getLogger(__name__)


class FINANCIAL_TIMES(Retrieval):
    """
    This class can be used to retrieve the excess mortality data from the Financial Times
    `github repository <https://github.com/Financial-Times/coronavirus-excess-mortality-data>`_.

    Example
    -------
    .. code-block::

        ft = cov19.data_retrieval.FINANCIAL_TIMES()
        ft.download_all_available_data()

        #Access the data by
        ft.data
        #or
        ft.get(filter) #see below
    """

    def __init__(self, auto_download=False):
        """
        On init of this class the base Retrieval Class __init__ is called, with financial
        times specific arguments.

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
        name = "Financial_times"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = "https://raw.githubusercontent.com/Financial-Times/coronavirus-excess-mortality-data/master/data/ft_excess_deaths.csv"

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Suppress warning

        """
        If the local file is older than the update_interval it gets updated once the
        download all function is called. Can be diffent values depending on the parent class
        """
        update_interval = datetime.timedelta(days=1)

        # Init the retrieval base class
        Retrieval.__init__(
            self,
            name,
            url_csv,
            [_data_dir_fallback + "/" + name + "_fallback.csv.gz"],
            update_interval,
            **kwargs,
        )

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
        """
        Converts the data to a usable format i.e. converts all date string to
        datetime objects and some other column names.

        This is most of the time the first place one has to look at if something breaks!

        self.data -> self.data converted
        """
        try:
            df = self.data
            # datetime columns
            df["date"] = pd.to_datetime(df["date"])
            df = df.rename(columns={"region": "state"})  # For consistency
            df = df.set_index("date")
            self.data = df
            return True
        except Exception as e:
            log.warning(f"There was an error formating the data! {e}")
            raise e
        return False

    def get(
        self,
        value="excess_deaths",
        country: str = "Germany",
        state: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
    ):
        """
        Retrieves specific data from the dataset, can be filtered by date, country and state.

        Parameters
        ----------
        value : str, optional
            Which data to return, possible values are
            - "deaths",
            - "expected_deaths",
            - "excess_deaths",
            - "excess_deaths_pct"
            (default: "excess_deaths")
        country : str, optional
        state : str, optional
            Possible countries and states can be retrieved by the `get_possible_countries_states()` method.
        begin_date : datetime.datetime, optional
            First day that should be filtered
        end_date : datetime.datetime, optional
            Last day that should be filtered
        """

        # ------------------------------------------------------------------------------ #
        # Default Parameters
        # ------------------------------------------------------------------------------ #
        possible_values = [
            "deaths",
            "expected_deaths",
            "excess_deaths_pct",
            "excess_deaths",
        ]
        assert (
            value in possible_values
        ), f"Value '{value}' not possible! Use one from {possible_values}"

        if state is None:
            state = country  # somehow they publish the data like that ¯\_(ツ)_/¯

        possible_countries_states = self.get_possible_countries_states()
        assert [
            country,
            state,
        ] in possible_countries_states, f"Country, state combination '[{country},{state}]' not possible! Check possible combinations by get_possible_countries_states()!"

        if data_begin is None:
            data_begin = self.__get_first_date()
        if data_end is None:
            data_end = self.__get_last_date()

        # ------------------------------------------------------------------------------ #
        # Filter the data
        # ------------------------------------------------------------------------------ #

        # Filter by country first
        df = self.data[self.data["country"] == country]

        # Filter by state next
        df = df[df["state"] == state]

        # Filter by value
        df = df[value]

        # Filter by date
        df = df[data_begin:data_end]

        return df

    def get_possible_countries_states(self):
        """
        Can be used to obtain all different possible countries with there corresponding possible states and regions.

        Returns
        -------
        : pandas.DataFrame
        """
        return self.data[["country", "state"]].drop_duplicates().to_numpy()

    def __get_first_date(self):
        return self.data.index.min()

    def __get_last_date(self):
        return self.data.index.max()
