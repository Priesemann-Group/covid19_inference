class RKIsituationreports:
    """
    As mentioned by Matthias Linden, the daily situation reports have more available data.
    This class retrieves this additional data from Matthias website and parses it into the format we use i.e. a datetime index.

    Interesting new data is for example ICU cases, deaths and recorded symptoms. For now one can look at the data by running

    .. code-block::

        rki_si_re = cov19.data_retrieval.RKIsituationreports(True)
        print(rki_si_re.data)

    ToDo
    -----
    Filter functions for ICU, Symptoms and maybe even daily new cases for the respective categories.

    """

    def __init__(self, auto_download=True):
        self.data: pd.DataFrame
        if auto_download:
            self.download_all_available_data()

    def download_all_available_data(self, url: str = None):
        """
        Attempts to download the most current data from Matthias Lindens website. Fallback to his github page and finaly a local copy.

        Parameters
        ----------
        url : str, optional
            Default filepath to the source

        Returns
        -------
        : pandas.DataFrame

        """
        if url is None:
            url = "http://mlinden.de/COVID19/data/latest_report.csv"

        # Path to the local fallback file
        this_dir = get_data_dir()
        fallback = this_dir + "/../data/rkisituationreport_fallback.csv.gz"

        # We can just download it every run since it is very small -> We have to do that anyways because to look at the date header
        df = self.__download_from_source(url, fallback)
        print(self.__to_iso(df))

        # Download file and overwrite old one if it is older
        self.data = df

        return self.data

    def __download_from_source(self, url, fallback=None):
        """
        Private method
        Downloads one csv file from an url and converts it into a pandas dataframe. A fallback source can also be given.

        Parameters
        ----------
        url : str
            Where to download the csv file from
        fallback : str, optional
            Fallback source for the csv file, filename of file that is located in /data/

        Returns
        -------
        : pandas.DataFrame
            Raw data from the source url as dataframe
        """

        try:
            data = pd.read_csv(url, sep=";", header=1)
        except Exception as e:
            log.info(
                "Failed to download current data 'confirmed cases', using local copy."
            )
            this_dir = get_data_dir()
            data = pd.read_csv(fallback, sep=";", header=1)
        # Save as local backup if newer
        data.to_csv(fallback, sep=";", header=1, compression="infer")
        return data

    def __to_iso(self, df):
        if "Unnamed: 0" in df.columns:
            df["date"] = pd.to_datetime(df["Unnamed: 0"])
            df = df.drop(columns="Unnamed: 0")
        df = df.set_index(["date"])
        return df
