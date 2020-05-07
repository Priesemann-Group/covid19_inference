class JHU:
    """
    This class can be used to retrieve and filter the dataset from the online repository of the coronavirus visual dashboard operated
    by the `Johns Hopkins University <https://coronavirus.jhu.edu/>`_.

    Features
        - download all files from the online repository of the coronavirus visual dashboard operated by the Johns Hopkins University.
        - filter by deaths, confirmed cases and recovered cases
        - filter by country and state
        - filter by date

    Parameters
    ----------
    auto_download : bool, optional
        whether or not to automatically download the data from jhu (default: false)

    Example
    -------
    .. code-block::
    
        jhu = cov19.data_retrieval.JHU()
        jhu.download_all_available_data()

        #Acess the data by
        jhu.data
        #or
        jhu.get_new(args)
        jhu.get_total(args)
    """

    @property
    def data(self):
        return (self.confirmed, self.deaths, self.recovered)

    def __init__(self, auto_download=True):
        self.confirmed: pd.DataFrame
        self.deaths: pd.DataFrame
        self.recovered: pd.DataFrame

        if auto_download:
            self.download_all_available_data()

    def download_all_available_data(
        self,
        fp_confirmed: str = None,
        fp_deaths: str = None,
        fp_recovered: str = None,
        save_to_attributes: bool = True,
    ):
        """
        Attempts to download the most current data for the confirmed cases, deaths and recovered cases from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University. If the repo is not available it should fallback to the local files located in /data/.
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        fp_confirmed,fp_deaths,fp_recovered : str, optional
            Filepath or URL pointing to the original CSV of global confirmed cases, deaths or recovered cases. Default download sources are
            `Confirmed <https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv>`_,
            `Deaths <https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv>`_ and
            `Recovered <https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv>`_. (default: None)

        save_to_attributes : bool, optional
            Should the returned dataframe tuple be saved as attributes (default:true)

        Returns
        -------
        : pandas.DataFrame tuple
            tuple of table with confirmed, deaths and recovered cases


        """
        if fp_confirmed is None:
            fp_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        if fp_deaths is None:
            fp_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
        if fp_recovered is None:
            fp_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

        # Fallbacks should be set automatically in the download_* functions

        return (
            self._download_confirmed(fp_confirmed, save_to_attributes),
            self._download_deaths(fp_deaths, save_to_attributes),
            self._download_recovered(fp_recovered, save_to_attributes),
        )

    def _download_confirmed(
        self,
        fp_confirmed: str = None,
        save_to_attributes: bool = True,
        fallback: str = None,
    ):
        """
        Attempts to download the most current data for the confirmed cases from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University. If the repo is not available it falls back to
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        fp_confirmed : str, optional
            Filepath or URL pointing to the original CSV of global confirmed cases, deaths or recovered cases
        save_to_attributes : bool, optional
            Should the returned dataframe be saved as attributes (default:true)
        fallback: str, optional
            Filepath to a fallback source, should be set automatically by default

        Returns
        -------
        : pandas.DataFrame
            Table with confirmed cases, indexed by date
        """
        if fp_confirmed is None:
            fp_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

        fallbacks = [
            get_data_dir() + "/jhu_fallback_confirmed.csv.gz",
        ]
        if fallback is not None:
            fallbacks.append(fallback)
        fallbacks.append(_data_dir_fallback + "/jhu_fallback_confirmed.csv.gz",)

        dl = self.__download_from_source(
            url=fp_confirmed, fallbacks=fallbacks, write_to=fallbacks[0],
        )
        log.debug(dl)
        confirmed = self.__to_iso(dl)
        if save_to_attributes:
            self.confirmed = confirmed
        return confirmed

    def _download_deaths(
        self,
        fp_deaths: str = None,
        save_to_attributes: bool = True,
        fallback: str = None,
    ):
        """
        Attempts to download the most current data for the deaths from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University.
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        fp_deaths : str, optional
            filepath or URL pointing to the original CSV of global confirmed cases, deaths or recovered cases
        save_to_attributes : bool, optional
            Should the returned dataframe be saved as attributes (default:true)
        fallback: str, optional
            Filepath to a fallback source, should be set automatically by default

        Returns
        -------
        : pandas.DataFrame
            Table with deaths, indexed by date
        """
        if fp_deaths is None:
            fp_deaths = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"

        fallbacks = [
            get_data_dir() + "/jhu_fallback_deaths.csv.gz",
        ]
        if fallback is not None:
            fallbacks.append(fallback)
        fallbacks.append(_data_dir_fallback + "/jhu_fallback_deaths.csv.gz",)

        deaths = self.__to_iso(
            self.__download_from_source(
                url=fp_deaths, fallbacks=fallbacks, write_to=fallbacks[0],
            )
        )

        if save_to_attributes:
            self.deaths = deaths
        return deaths

    def _download_recovered(
        self,
        fp_recovered: str = None,
        save_to_attributes: bool = True,
        fallback: str = None,
    ):
        """
        Attempts to download the most current data for the recovered cases from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University.
        Only works if the module is located in the repo directory.

        Parameters
        ----------
        fp_recovered : str, optional
            Filepath or URL pointing to the original CSV of global confirmed cases, deaths or recovered cases
        save_to_attributes : bool, optional
            Should the returned dataframe be saved as attributes (default:true)
        fallback: str, optional
            Filepath to a fallback source, should be set automatically by default

        Returns
        -------
        : pandas.DataFrame
            Table with recovered cases, indexed by date
        """
        if fp_recovered is None:
            fp_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

        fallbacks = [
            get_data_dir() + "/jhu_fallback_recovered.csv.gz",
        ]
        if fallback is not None:
            fallbacks.append(fallback)
        fallbacks.append(_data_dir_fallback + "/jhu_fallback_recovered.csv.gz",)

        recovered = self.__to_iso(
            self.__download_from_source(
                url=fp_recovered, fallbacks=fallbacks, write_to=fallbacks[0],
            )
        )

        if save_to_attributes:
            self.recovered = recovered

        return recovered

    def __download_from_source(self, url, fallbacks=[], write_to=None):
        """
        Private method
        Downloads one csv file from an url and converts it into a pandas dataframe. A fallback source can also be given.

        Parameters
        ----------
        url : str
            Where to download the csv file from
        fallbacks : list, optional
            List of optional fallback sources for the csv file.
        write_to : str, optional
            If provided, save the downloaded_data there. Default: None, do not write.

        Returns
        -------
        : pandas.DataFrame
            Raw data from the source url as dataframe
        """
        try:
            data = pd.read_csv(url, sep=",")
            data["Country/Region"] = iso_3166_convert_to_iso(
                data["Country/Region"]
            )  # convert before saving so we do not have to do it every time
            # Save to write_to. A bit hacky but should work
            if write_to is not None:
                data.to_csv(write_to, sep=",", index=False, compression="infer")
        except Exception as e:
            log.info(
                f"Failed to download {url}: {e}, trying {len(fallbacks)} fallbacks."
            )
            for fb in fallbacks:
                try:
                    data = pd.read_csv(fb, sep=",")
                    # so this was successfull, make a copy
                    if write_to is not None:
                        data.to_csv(write_to, sep=",", index=False, compression="infer")
                    log.debug(f"Fallback {fb} successful.")
                    break
                except Exception as e:
                    continue
        return data

    def __to_iso(self, df):
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

        df = df.drop(columns=["Lat", "Long"]).rename(
            columns={"Province/State": "state", "Country/Region": "country"}
        )
        df = df.set_index(["country", "state"])
        df.columns = pd.to_datetime(df.columns)

        # datetime columns
        return df.T

    def get_total_confirmed_deaths_recovered(
        self,
        country: str = None,
        state: str = None,
        begin_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
    ):
        """
        Retrieves all confirmed, deaths and recovered cases from the Johns Hopkins University dataset as a DataFrame with datetime index.
        Can be filtered by country and state, if only a country is given all available states get summed up.

        Parameters
        ----------
        country : str, optional
            name of the country (the "Country/Region" column), can be None if the whole summed up data is wanted (why would you do this?)
        state : str, optional
            name of the state (the "Province/State" column), can be None if country is set or the whole summed up data is wanted
        begin_date : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used
        end_date : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used

        Returns
        -------
        : pandas.DataFrame
        """

        # filter
        df = pd.DataFrame(
            columns=["date", "confirmed", "deaths", "recovered"]
        ).set_index("date")
        if country is None:
            df["confirmed"] = self.confirmed.sum(axis=1, skipna=True)
            df["deaths"] = self.deaths.sum(axis=1, skipna=True)
            df["recovered"] = self.recovered.sum(axis=1, skipna=True)
        else:
            if state is None:
                df["confirmed"] = self.confirmed[country].sum(axis=1, skipna=True)
                df["deaths"] = self.deaths[country].sum(axis=1, skipna=True)
                df["recovered"] = self.recovered[country].sum(axis=1, skipna=True)
            else:
                df["confirmed"] = self.confirmed[(country, state)]
                df["deaths"] = self.deaths[(country, state)]
                df["recovered"] = self.recovered[(country, state)]
        df.index.name = "date"

        return self.filter_date(df, begin_date, end_date)

    def get_new(
        self,
        value="confirmed",
        country: str = None,
        state: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
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

        Returns
        -------
        : pandas.DataFrame
            table with new cases and the date as index

        """

        # ------------------------------------------------------------------------------ #
        # Default Parameters
        # ------------------------------------------------------------------------------ #
        if value not in ["confirmed", "recovered", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        if country == "None":
            country = None
        if state == "None":
            state = None

        # If no date is given set to first and last dates in data
        if data_begin is None:
            data_begin = self.__get_first_date()
        if data_end is None:
            data_end = self.__get_last_date()

        if data_begin == self.data[0].index[0]:
            raise ValueError("Date has to be after the first dataset entry")

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #
        df = pd.DataFrame(columns=["date", value]).set_index("date")
        if country is None:
            df[value] = getattr(self, value).sum(axis=1, skipna=True)
        else:
            if state is None:
                df[value] = getattr(self, value)[country].sum(axis=1, skipna=True)
            else:
                df[value] = getattr(self, value)[(country, state)]
        df.index.name = "date"

        df = self.filter_date(df, data_begin - datetime.timedelta(days=1), data_end)
        df = (
            df.diff().drop(df.index[0]).astype(int)
        )  # Neat oneliner to also drop the first row and set the type back to int
        return df

    def get_total(
        self,
        value="confirmed",
        country: str = None,
        state: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
    ):
        """
        Retrieves all total/cumulative cases from the Johns Hopkins University dataset as a DataFrame with datetime index.
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

        Returns
        -------
        : pandas.DataFrame
            table with total/cumulative cases and the date as index
        """

        # ------------------------------------------------------------------------------ #
        # Default Parameters
        # ------------------------------------------------------------------------------ #
        if value not in ["confirmed", "recovered", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        if country == "None":
            country = None
        if state == "None":
            state = None

        # Note: It should be fine to NOT check for the date since this is also done by the filter_date method

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #
        df = pd.DataFrame(columns=["date", value]).set_index("date")
        if country is None:
            df[value] = getattr(self, value).sum(axis=1, skipna=True)
        else:
            if state is None:
                df[value] = getattr(self, value)[country].sum(axis=1, skipna=True)
            else:
                df[value] = getattr(self, value)[(country, state)]
        df.index.name = "date"
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
            begin_date = self.__get_first_date(df)
        if end_date is None:
            end_date = self.__get_last_date(df)

        if not isinstance(begin_date, datetime.datetime) and isinstance(
            end_date, datetime.datetime
        ):
            raise ValueError(
                "Invalid begin_date, end_date: has to be datetime.datetime object"
            )

        return df[begin_date:end_date]

    def __get_first_date(self):
        return df.data[0].index[0]

    def __get_last_date(self):
        return df.data[0].index[-1]

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
