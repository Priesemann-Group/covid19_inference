class RKI:
    """
    This class can be used to retrive and filter the dataset from the Robert Koch Institute `Robert Koch Institute <https://www.rki.de/>`_.
    The data gets retrieved from the `arcgis <https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data>`_  dashboard.

    Features
        - download the full dataset
        - filter by date
        - filter by bundesland
        - filter by recovered, deaths and confirmed cases

    Parameters
    ----------
    auto_download : bool, optional
        whether or not to automatically download the data from rki (default: false)

    Example
    -------
    .. code-block::
    
        rki = cov19.data_retrieval.RKI()
        rki.download_all_available_data()

        #Acess the data by
        rki.data
        #or
        rki.get_new("confirmed","Sachsen")
        rki.get_total(args)    
    """

    def __init__(self, auto_download=False):
        self.data: pd.DataFrame

        if auto_download:
            self.download_all_available_data()

    def download_all_available_data(self):
        """
        Attempts to download the most current data from the Robert Koch Institute. Separated into the different regions (landkreise).

        Parameters
        ----------
        try_max : int, optional
            Maximum number of tries for each query. (default:10)

        Returns
        -------
        : pandas.DataFrame
            Containing all the RKI data from arcgis website.
            In the format:
            [Altersgruppe, AnzahlFall, AnzahlGenesen, AnzahlTodesfall, Bundesland, Geschlecht, Landkreis, Meldedatum, NeuGenesen, NeuerFall, Refdatum, date, date_ref, Datenstand, ]

        """

        # We need an extra url since for some reason the normal dataset website has no headers :/ --> But they get updated from the same source so that should work
        url_fulldata = "https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data"

        url_check_update = "https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=0%3D0&objectIds=&time=&resultType=none&outFields=Datenstand&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=true&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token="

        # Path to the local file, where we want the data in the end
        url_local = get_data_dir() + "/rki_fallback.csv.gz"

        # where to look for already downloaded content
        fallbacks = [
            get_data_dir() + "/rki_fallback.csv.gz",
            _data_dir_fallback + "/rki_fallback.csv.gz",
        ]

        # Loads local copy and gets latest data date
        df = None

        for fb in fallbacks:
            try:
                log.debug(f"Trying local file {fb}")
                # Local copy should be properly formated, so no __to_iso() used
                df = pd.read_csv(fb, sep=",")
                current_file_date = datetime.datetime.strptime(
                    df.Datenstand.unique()[0], "%d.%m.%Y, %H:%M Uhr"
                )
                break
            except Exception as e:
                log.debug(f"Local file not available: {e}")
                current_file_date = datetime.datetime.fromtimestamp(0)

        # Get last modified date for the files from rki repo
        try:
            with urllib.request.urlopen(url_check_update) as url:
                json_data = json.loads(url.read().decode())
            if len(json_data["features"]) > 1:
                raise RuntimeError(
                    "Date checking file has more than one Datenstand. "
                    + "This should not happen."
                )
            online_file_date = datetime.datetime.strptime(
                json_data["features"][0]["attributes"]["Datenstand"],
                "%d.%m.%Y, %H:%M Uhr",
            )
        except Exception as e:
            log.debug("Could not fetch data date from online repository of the RKI")
            online_file_date = datetime.datetime.fromtimestamp(1)

        # Download file and overwrite old one if it is older
        if online_file_date > current_file_date:
            log.info("Downloading rki dataset from repository.")
            try:
                df = self.__to_iso(pd.read_csv(url_fulldata, sep=","))
            except Exception as e:
                log.warning(
                    "Download Failed! Trying downloading via rest api. May take longer!"
                )
                log.warning(e)
                try:
                    # Dates already are datetime, so no __to_iso used
                    df = self.__download_via_rest_api(try_max=10)
                except Exception as e:
                    log.warning("Downloading from the rest api also failed!")

                    if df is None:
                        raise RuntimeError("No source to obtain RKI data from.")

            log.debug(f"Overwriting {url_local} with newest downloaded data.")
            df.to_csv(url_local, compression="infer", index=False)
        else:
            log.info("Using local rki data because no newer version available online.")
            for fb in fallbacks:
                try:
                    df = pd.read_csv(fb, sep=",")
                    if fb != url_local:
                        df.to_csv(url_local, compression="infer", index=False)
                    break
                except Exception as e:
                    log.debug(f"{e}")

        self.data = df

        return df

    def __to_iso(self, df) -> pd.DataFrame:
        if "Meldedatum" in df.columns:
            df["date"] = df["Meldedatum"].apply(
                lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
            )
            df = df.drop(columns="Meldedatum")
        if "Refdatum" in df.columns:
            df["date_ref"] = df["Refdatum"].apply(
                lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
            )
            df = df.drop(columns="Refdatum")

        # Rename the columns to match the JHU dataset
        if "AnzahlFall" in df.columns:
            df.rename(columns={"AnzahlFall": "confirmed"}, inplace=True)
        if "AnzahlTodesfall" in df.columns:
            df.rename(columns={"AnzahlTodesfall": "deaths"}, inplace=True)
        if "AnzahlGenesen" in df.columns:
            df.rename(columns={"AnzahlGenesen": "recovered"}, inplace=True)

        df["date"] = pd.to_datetime(df["date"])
        df["date_ref"] = pd.to_datetime(df["date_ref"])
        return df

    def __download_via_rest_api(self, try_max=10):
        landkreise_max = 412  # Strangely there are 412 regions defined by the Robert Koch Insitute in contrast to the offical 294 rural districts or the 401 administrative districts.
        url_id = "https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0/query?where=0%3D0&objectIds=&time=&resultType=none&outFields=idLandkreis&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=true&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token="

        url = urllib.request.urlopen(url_id)
        json_data = json.loads(url.read().decode())
        n_data = len(json_data["features"])
        unique_ids = [
            json_data["features"][i]["attributes"]["IdLandkreis"] for i in range(n_data)
        ]

        # If the number of landkreise is smaller than landkreise_max, uses local copy (query system can behave weirdly during updates)
        if n_data >= landkreise_max:
            log.info(f"Downloading {n_data} unique Landkreise. May take a while.\n")
            df_keys = [
                "IdBundesland",
                "Bundesland",
                "Landkreis",
                "Altersgruppe",
                "Geschlecht",
                "AnzahlFall",
                "AnzahlTodesfall",
                "ObjectId",
                "IdLandkreis",
                "Datenstand",
                "NeuerFall",
                "NeuerTodesfall",
                "NeuGenesen",
                "AnzahlGenesen",
                "date",
                "date_ref",
            ]

            df = pd.DataFrame(columns=df_keys)

            # Fills DF with data from all landkreise
            for idlandkreis in unique_ids:

                url_str = (
                    "https://services7.arcgis.com/mOBPykOjAyBO2ZKk/ArcGIS/rest/services/RKI_COVID19/FeatureServer/0//query?where=IdLandkreis%3D"
                    + idlandkreis
                    + "&objectIds=&time=&resultType=none&outFields=Bundesland%2C+Landkreis%2C+IdBundesland%2C+ObjectId%2C+IdLandkreis%2C+Altersgruppe%2C+Geschlecht%2C+AnzahlFall%2C+AnzahlTodesfall%2C+Meldedatum%2C+NeuerFall%2C+Refdatum%2C+Datenstand%2C+NeuGenesen%2C+AnzahlGenesen&returnIdsOnly=false&returnUniqueIdsOnly=false&returnCountOnly=false&returnDistinctValues=false&cacheHint=false&orderByFields=&groupByFieldsForStatistics=&outStatistics=&having=&resultOffset=&resultRecordCount=&sqlFormat=none&f=pjson&token="
                )

                count_try = 0

                while count_try < try_max:
                    try:
                        with urllib.request.urlopen(url_str) as url:
                            json_data = json.loads(url.read().decode())

                        n_data = len(json_data["features"])

                        if n_data > 5000:
                            raise ValueError("Query limit exceeded")

                        data_flat = [
                            json_data["features"][i]["attributes"]
                            for i in range(n_data)
                        ]

                        break

                    except:
                        count_try += 1

                if count_try == try_max:
                    raise ValueError("Maximum limit of tries exceeded.")

                df_temp = pd.DataFrame(data_flat)

                # Very inneficient, but it will do
                df = pd.concat([df, df_temp], ignore_index=True)

            df["date"] = df["Meldedatum"].apply(
                lambda x: datetime.datetime.fromtimestamp(x / 1e3)
            )
            df["date_ref"] = df["Refdatum"].apply(
                lambda x: datetime.datetime.fromtimestamp(x / 1e3)
            )
            df = df.drop(columns="Meldedatum")
            df = df.drop(columns="Refdatum")

        else:
            raise RuntimeError("Invalid response from REST api")

        return df

    def get_total(
        self,
        value="confirmed",
        bundesland: str = None,
        landkreis: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
        date_type: str = "date",
    ):
        """
        Gets all total confirmed cases for a region as dataframe with date index. Can be filtered with multiple arguments.

        Parameters
        ----------
        value: str
            Which data to return, possible values are 
            - "confirmed",
            - "recovered",
            - "deaths"
            (default: "confirmed")
        bundesland : str, optional
            if no value is provided it will use the full summed up dataset for Germany
        landkreis : str, optional
            if no value is provided it will use the full summed up dataset for the region (bundesland)
        data_begin : datetime.datetime, optional
            initial date, if no value is provided it will use the first possible date
        data_end : datetime.datetime, optional
            last date, if no value is provided it will use the most recent possible date
        date_type : str, optional
            type of date to use: reported date 'date' (Meldedatum in the original dataset), or symptom date 'date_ref' (Refdatum in the original dataset)

        Returns
        -------
        :pandas.DataFrame
        """

        # ------------------------------------------------------------------------------ #
        # Default parameters
        # ------------------------------------------------------------------------------ #
        if value not in ["confirmed", "recovered", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        # Note: It should be fine to NOT check for the date since this is also done by the filter_date method

        # Set level for filter use bundesland if no landkreis is supplied else use landkreis
        level = None
        filter_value = None
        if bundesland is not None and landkreis is None:
            level = "Bundesland"
            filter_value = bundesland
        elif bundesland is None and landkreis is not None:
            level = "Landkreis"
            filter_value = landkreis
        elif bundesland is not None and landkreis is not None:
            raise ValueError("bundesland and landkreis cannot be simultaneously set.")

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #
        df = self.filter(data_begin, data_end, value, date_type, level, filter_value)
        return df

    def get_new(
        self,
        value="confirmed",
        bundesland: str = None,
        landkreis: str = None,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
        date_type: str = "date",
    ):
        """
        Retrieves all new cases from the Robert Koch Institute dataset as a DataFrame with datetime index.
        Can be filtered by value, bundesland and landkreis, if only a country is given all available states get summed up.

        Parameters
        ----------
        value: str
            Which data to return, possible values are 
            - "confirmed",
            - "recovered",
            - "deaths"
            (default: "confirmed")
        bundesland : str, optional
            if no value is provided it will use the full summed up dataset for Germany
        landkreis : str, optional
            if no value is provided it will use the full summed up dataset for the region (bundesland)
        data_begin : datetime.datetime, optional
            intial date for the returned data, if no value is given the first date in the dataset is used,
            if none is given could yield errors
        data_end : datetime.datetime, optional
            last date for the returned data, if no value is given the most recent date in the dataset is used

        Returns
        -------
        : pandas.DataFrame
            table with daily new confirmed and the date as index
        """

        # ------------------------------------------------------------------------------ #
        # Default parameters
        # ------------------------------------------------------------------------------ #

        if value not in ["confirmed", "recovered", "deaths"]:
            raise ValueError(
                'Invalid value. Valid options: "confirmed", "deaths", "recovered"'
            )

        level = None
        filter_value = None
        if bundesland is not None and landkreis is None:
            level = "Bundesland"
            filter_value = bundesland
        elif bundesland is None and landkreis is not None:
            level = "Landkreis"
            filter_value = landkreis
        elif bundesland is not None and landkreis is not None:
            raise ValueError("bundesland and landkreis cannot be simultaneously set.")

        if data_begin is None:
            data_begin = self.data[date_type].iloc[0]
        if data_end is None:
            data_end = self.data[date_type].iloc[-1]

        if data_begin == self.data[date_type].iloc[0]:
            raise ValueError(
                "Date has to be after the first dataset entry. Set a data_begin date!"
            )

        # ------------------------------------------------------------------------------ #
        # Retrieve data and filter it
        # ------------------------------------------------------------------------------ #

        df = self.filter(
            data_begin - datetime.timedelta(days=1),
            data_end,
            value,
            date_type,
            level,
            filter_value,
        )
        # Get difference to the days beforehand
        df = (
            df.diff().drop(df.index[0]).astype(int)
        )  # Neat oneliner to also drop the first row and set the type back to int
        return df.fillna(0)

    def filter(
        self,
        data_begin: datetime.datetime = None,
        data_end: datetime.datetime = None,
        variable="confirmed",
        date_type="date",
        level=None,
        value=None,
    ):
        """
        Filters the obtained dataset for a given time period and returns an array ONLY containing only the desired variable.

        Parameters
        ----------
        data_begin : datetime.datetime, optional
            initial date, if no value is provided it will use the first possible date
        data_end : datetime.datetime, optional
            last date, if no value is provided it will use the most recent possible date
        variable : str, optional
            type of variable to return
            possible types are:
            "confirmed"      : cases (default)
            "AnzahlTodesfall" : deaths
            "AnzahlGenesen"   : recovered
        date_type : str, optional
            type of date to use: reported date 'date' (Meldedatum in the original dataset), or symptom date 'date_ref' (Refdatum in the original dataset)
        level : str, optional
            possible strings are:
                "None"       : return data from all Germany (default)
                "Bundesland" : a state
                "Landkreis"  : a region
        value : None, optional
            string of the state/region
            e.g. "Sachsen"

        Returns
        -------
        : pd.DataFrame
            array with ONLY the requested variable, in the requested range. (one dimensional)
        """
        # Input parsing
        if variable not in ["confirmed", "deaths", "recovered"]:
            raise ValueError(
                'Invalid variable. Valid options: "confirmed", "deaths", "recovered"'
            )

        if level not in ["Landkreis", "Bundesland", None]:
            raise ValueError(
                'Invalid level. Valid options: "Landkreis", "Bundesland", None'
            )

        if date_type not in ["date", "date_ref"]:
            raise ValueError('Invalid date_type. Valid options: "date", "date_ref"')

        df = self.data.sort_values(date_type)
        if data_begin is None:
            data_begin = df[date_type].iloc[0]
        if data_end is None:
            data_end = df[date_type].iloc[-1]
        if not isinstance(data_begin, datetime.datetime) and isinstance(
            data_end, datetime.datetime
        ):
            raise ValueError(
                "Invalid data_begin, data_end: has to be datetime.datetime object"
            )

        # Keeps only the relevant data
        df = self.data

        if level is not None:
            df = df[df[level] == value][[date_type, variable]]

        df_series = df.groupby(date_type)[variable].sum().cumsum()
        df_series.index = pd.to_datetime(df_series.index)

        return df_series[data_begin:data_end].fillna(0)

    def filter_all_bundesland(
        self,
        begin_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        variable="confirmed",
        date_type="date",
    ):
        """
        Filters the full RKI dataset

        Parameters
        ----------
        df : DataFrame
            RKI dataframe, from get_rki()
        begin_date : datetime.datetime
            initial date to return
        end_date : datetime.datetime
            last date to return
        variable : str, optional
            type of variable to return: cases ("AnzahlFall"), deaths ("AnzahlTodesfall"), recovered ("AnzahlGenesen")
        date_type : str, optional
            type of date to use: reported date 'date' (Meldedatum in the original dataset), or symptom date 'date_ref' (Refdatum in the original dataset)

        Returns
        -------
        : pd.DataFrame
            DataFrame with datetime dates as index, and all German regions (bundesländer) as columns
        """
        if variable not in ["confirmed", "deaths", "recovered"]:
            raise ValueError(
                'Invalid variable. Valid options: "confirmed", "deaths", "recovered"'
            )

        if date_type not in ["date", "date_ref"]:
            raise ValueError('Invalid date_type. Valid options: "date", "date_ref"')

        if begin_date is None:
            begin_date = self.data[date_type].iloc[0]
        if end_date is None:
            end_date = self.data[date_type].iloc[-1]

        if not isinstance(begin_date, datetime.datetime) and isinstance(
            end_date, datetime.datetime
        ):
            raise ValueError(
                "Invalid begin_date, end_date: has to be datetime.datetime object"
            )

        # Nifty, if slightly unreadable one-liner
        df = self.data
        df2 = (
            df.groupby([date_type, "Bundesland"])[variable]
            .sum()
            .reset_index()
            .pivot(index=date_type, columns="Bundesland", values=variable)
            .fillna(0)
        )
        df2.index = pd.to_datetime(df2.index)
        # Returns cumsum of variable
        return df2[begin_date:end_date].cumsum()
