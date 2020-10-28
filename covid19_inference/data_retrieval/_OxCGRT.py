import datetime
import pandas as pd
import logging

# Import base class
from .retrieval import Retrieval, _data_dir_fallback

log = logging.getLogger(__name__)


class OxCGRT(Retrieval):
    """
    This class can be used to retrieve the datasset on goverment policies from the
    `Oxford Covid-19 Government Response Tracker <https://github.com/OxCGRT/covid-policy-tracker>`_.


    Example
    -------
    .. code-block::

        gov_pol = cov19.data_retrieval.OxCGRT()
        gov_pol.download_all_available_data()

    """

    def __init__(self, auto_download=False):
        """
        On init of this class the base Retrieval Class __init__ is called, with google specific
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
        name = "OxCGRT"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = "https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv"

        """
        Kwargs for pandas read csv
        """
        kwargs = {}  # Surpress warning

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
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            if "CountryName" in df.columns:
                df = df.rename(columns={"CountryName": "country"})
            # datetime columns
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df = df.set_index("date")
            self.data = df
        except Exception as e:
            log.warning(f"There was an error formating the data! {e}")
            raise e
        return False

    def get_possible_countries(self):
        """
            Can be used to obtain all different possible countries in the dataset.

            Returns
            -------
            : pandas.DataFrame
        """
        return self.data["country"].unique()

    def get_possible_policies(self):
        """
            Can be used to obtain all policies in there corresponding categories possible countries in the dataset.

            Returns
            -------
            : dict
        """

        ret = dict()
        ret["containment and closure policies"] = []
        ret["economic policies"] = []
        ret["health system policies"] = []
        ret["miscellaneous policies"] = []

        for policy in self.data.columns:
            if (
                policy.startswith("C")
                and any(map(str.isdigit, policy))
                and policy[-4:] != "Flag"
            ):
                ret["containment and closure policies"].append(policy)

            if (
                policy.startswith("E")
                and any(map(str.isdigit, policy))
                and policy[-4:] != "Flag"
            ):
                ret["economic policies"].append(policy)

            if (
                policy.startswith("H")
                and any(map(str.isdigit, policy))
                and policy[-4:] != "Flag"
            ):
                ret["health system policies"].append(policy)

            if (
                policy.startswith("M")
                and any(map(str.isdigit, policy))
                and policy[-4:] != "Flag"
            ):
                ret["miscellaneous policies"].append(policy)

        return ret

    def get_change_points(self, policies, country):
        """
            Returns a list of change points, depending on the selected measure and country.

            Parameters
            ----------
            policies : str, array of str
                The wanted policies. Can be an array of strings, use get_possible_policies() to get
                a dict of possible policies.

            country : str
                Filter for country, use get_possible_countries() to get a list of possible ones.

            Returns
            -------
            :array of dicts
        """

        if isinstance(policies, str):
            policies = [policies]

        change_points = []

        # 1. Select by country
        df = self.data[self.data["country"] == country]

        # 2. Select each policy
        for policy in policies:
            df_t = df[policy].dropna(axis=0)

            # 3. Iterate over every date and check if the last value changed
            value_before = 0
            for date, value in df_t.iteritems():
                if value_before != value:
                    log.debug(f"Change point found:\n\t{policy}\n\t{date}")
                    cp_temp = dict(
                        date=date,
                        policy=policy,
                        indicator_before=value_before,
                        indicator_after=value,
                    )
                    change_points.append(cp_temp)
                value_before = value

        return change_points

    def get_time_data(self, policy, country, data_begin=None, data_end=None):
        """
            Parameters
            ----------
            policy : str
                The wanted policy.
            country : str
                Filter for country, use get_possible_countries() to get a list of possible ones.
            data_begin : datetime.datetime, optional
                intial date for the returned data, if no value is given the first date in the dataset is used,
                if none is given could yield errors
            data_end : datetime.datetime, optional
                last date for the returned data, if no value is given the most recent date in the dataset is used

            Returns
            -------
            :
                Pandas dataframe with policy
        """
        if data_begin is None:
            data_begin = self.__get_first_date()
        if data_end is None:
            data_end = self.__get_last_date()

        # 1. Select by country
        df = self.data[self.data["country"] == country]
        df_t = df[policy].dropna(axis=0)
        ix = pd.date_range(data_begin, data_end)
        return df_t[data_begin:data_end].reindex(ix)

    def __get_first_date(self):
        return self.data.index.min()

    def __get_last_date(self):
        return self.data.index.max()
