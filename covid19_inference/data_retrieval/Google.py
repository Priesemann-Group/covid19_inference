class GOOGLE:
    """
    `Google mobility data <https://www.google.com/covid19/mobility/>`_



    """

    def __init__(self, auto_download=False):
        self.data: pd.DataFrame
        if auto_download:
            self.download_all_available_data()

    def download_all_available_data(self, url: str = None):
        """
        Attempts to download the most current data from the Google mobility report.

        Parameters
        ----------
        try_max : int, optional
            Maximum number of tries for each query. (default:10)

        Returns
        -------
        : pandas.DataFrame

        """
        if url is None:
            url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"

        # Path to the local file where we want it in the end
        url_local = get_data_dir() + "/google_fallback.csv.gz"

        fallbacks = [
            get_data_dir() + "/google_fallback.csv.gz",
            _data_dir_fallback + "/google_fallback.csv.gz",
        ]

        # Get last modified dates for the files
        conn = urllib.request.urlopen(url, timeout=30)
        online_file_date = datetime.datetime.strptime(
            conn.headers["last-modified"].split(",")[-1], " %d %b %Y %H:%M:%S GMT"
        )

        try:
            current_file_date = datetime.datetime.fromtimestamp(
                os.path.getmtime(url_local)
            )
        except:
            current_file_date = datetime.datetime.fromtimestamp(2)

        # Download file and overwrite old one if it is older
        if online_file_date > current_file_date:
            log.info("Downloading new Google dataset from repository.")
            df = self.__to_iso(
                self.__download_from_source(
                    url=url, fallbacks=fallbacks, write_to=url_local
                )
            )
        else:
            log.info("Using local file since no new data is available online.")
            df = self.__to_iso(pd.read_csv(url_local, sep=",", low_memory=False))

        self.data = df

        return self.data

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
            data["country_region"] = iso_3166_convert_to_iso(
                data["country_region"]
            )  # convert before saving so we do not have to do it every time
            if write_to is not None:
                data.to_csv(write_to, sep=",", index=False, compression="infer")
        except Exception as e:
            log.info(
                f"Failed to download {url}: '{e}', trying {len(fallbacks)} fallbacks."
            )
            for fb in fallbacks:
                try:
                    data = pd.read_csv(fb, sep=",", low_memory=False)
                    # so this was successfull, make a copy
                    if write_to is not None:
                        data.to_csv(write_to, sep=",", index=False, compression="infer")
                    break
                except Exception as e:
                    continue
        log.info("Converting file to iso format, will take time, it is a huge dataset.")
        return data

    def __to_iso(self, df):
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
