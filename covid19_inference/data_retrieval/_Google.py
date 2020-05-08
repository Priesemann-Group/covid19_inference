import datetime
import pandas as pd

# Import base class
from .data_retrieval import Retrieval


class GOOGLE(Retrieval):
    """
    Google mobility data <https://www.google.com/covid19/mobility/>`_
    """

    def __init__(self, auto_download=False):
        # ------------------------------------------------------------------------------ #
        #  Init Data Base Class
        # ------------------------------------------------------------------------------ #

        # ------------------------------------------------------------------------------ #
        #  Init Retrieval Base Class
        # ------------------------------------------------------------------------------ #
        """
        A name mainly used for the Local Filename
        """
        name = "Google"

        """
        The url to the main dataset as csv, if none if supplied the fallback routines get used
        """
        url_csv = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

        """
        Kwargs for pandas read csv
        """
        kwargs = {"low_memory": False}  # Surpress warning

        # Init the retrieval base class
        Retrieval.__init__(self, name, url_csv, [], auto_download, **kwargs)

    def _to_iso(self, df):
        # change columns & index
        if (
            "country_region" in df.columns
            and "sub_region_1" in df.columns
            and "sub_region_2" in df.columns
        ):
            df = df.rename(
                columns={
                    "country_region": "country",
                    "sub_region_1": "state",
                    "sub_region_2": "region",
                }
            )
        df = df.set_index(["country", "state", "region"])
        # datetime columns
        df["date"] = pd.to_datetime(df["date"])
        return df

    def download_all_available_data(self, force_local=False, force_download=False):
        Retrieval.download_all_available_data(self, force_local, force_download)
        self._to_iso(self.data)

    def get_changes(
        self,
        country: str,
        state: str = None,
        region: str = None,
        begin_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
    ):
        """
        Returns a dataframe with the relative changes in mobility to a baseline, provided by google.
        They are separated into "retail and recreation", "grocery and pharmacy", "parks", "transit", "workplaces" and "residental".
        Filterable for country, state and region and date.

        Parameters
        ----------
        country : str
            Selected country for the mobility data.
        state : str, optional
            State for the selected data if no value is selected the whole country is chosen
        region : str, optional
            Region for the selected data if  no value is selected the whole region/country is chosen
        begin_date, end_date : datetime.datetime, optional
            Filter for the desired time period

        Returns
        -------
        : pandas.DataFrame
        """
        if country not in self.data.index:
            raise ValueError("Invalid country!")
        if state not in self.data.index and state is not None:
            raise ValueError("Invalid state!")
        if region not in self.data.index and region is not None:
            raise ValueError("Invalid region!")
        if begin_date is not None and isinstance(begin_date, datetime.datetime):
            raise ValueError("Invalid begin_date!")
        if end_date is not None and isinstance(end_date, datetime.datetime):
            raise ValueError("Invalid end_date!")

        # Select everything with that country
        if state is None:
            df = self.data.iloc[self.data.index.get_level_values("region").isnull()]
        else:
            df = self.data.iloc[self.data.index.get_level_values("region") == region]

        if state is None:
            df = df.iloc[df.index.get_level_values("state").isnull()]
        else:
            df = df.iloc[df.index.get_level_values("state") == state]

        df = df.iloc[df.index.get_level_values("country") == country]

        df = df.set_index("date")

        return df.drop(columns=["country_region_code"])[begin_date:end_date]

    def get_possible_counties_states_regions(self):
        """
        Can be used to obtain all different possible countries with there corresponding possible states and regions.

        Returns
        -------
        : pandas.DataFrame
        """
        return self.data.index.unique()
