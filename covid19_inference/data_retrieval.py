import datetime
import os
import logging
import tempfile
import platform
import stat

import numpy as np
import pandas as pd

import urllib, json

log = logging.getLogger(__name__)
# set by user, or default temp
_data_dir = None
# provided with the module
_data_dir_fallback = os.path.normpath(os.path.dirname(__file__) + "/../data/")

_format_date = lambda date_py: "{}/{}/{}".format(
    date_py.month, date_py.day, str(date_py.year)[2:4]
)


def set_data_dir(fname=None, permissions=None):
    """
        Set the global variable _data_dir. New downloaded data is placed there.
        If no argument provided we try the default tmp directory.
        If permissions are not provided, uses defaults if fname is in user folder.
        If not in user folder, tries to set 777.
    """

    target = "/tmp" if platform.system() == "Darwin" else tempfile.gettempdir()

    if fname is None:
        fname = f"{target}/covid19_data"
    else:
        try:
            fname = os.path.abspath(os.path.expanduser(fname))
        except Exception as e:
            log.debug("Specified file name caused an exception, using default")
            fname = f"{target}/covid19_data"

    log.debug(f"Setting global target directory to {fname}")
    fname += "/"
    os.makedirs(fname, exist_ok=True)

    try:
        log.debug(
            f"Trying to set permissions of {fname} "
            + f"({oct(os.stat(fname)[stat.ST_MODE])[-3:]}) "
            + f"to {'defaults' if permissions is None else str(permissions)}"
        )
        dirusr = os.path.abspath(os.path.expanduser("~"))
        if permissions is None:
            if not fname.startswith(dirusr):
                os.chmod(fname, 0o777)
        else:
            os.chmod(fname, int(str(permissions), 8))
    except Exception as e:
        log.debug(f"Unable set permissions of {fname}")

    global _data_dir
    _data_dir = fname
    log.debug(f"Target directory set to {_data_dir}")
    log.debug(f"{fname} (now) has permissions {oct(os.stat(fname)[stat.ST_MODE])[-3:]}")


def get_data_dir():
    if _data_dir is None or not os.path.exists(_data_dir):
        set_data_dir()
    return _data_dir


def iso_3166_add_alternative_name_to_iso_list(
    country_in_iso_3166: str, alternative_name: str
):
    this_dir = get_data_dir()
    try:
        data = json.load(open(this_dir + "/iso_countries.json", "r"))
    except Exception as e:
        data = json.load(open(_data_dir_fallback + "/iso_countries.json", "r"))

    try:
        data[country_in_iso_3166].append(alternative_name)
        log.info("Added alternative '{alternative_name}' to {country_in_iso_3166}.")
    except Exception as e:
        raise e

    json.dump(
        data,
        open(this_dir + "/iso_countries.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=4,
    )


def iso_3166_convert_to_iso(country_column_df):
    country_column_df = country_column_df.apply(
        lambda x: x
        if iso_3166_country_in_iso_format(x)
        else iso_3166_get_country_name_from_alternative(x)
    )
    return country_column_df


def iso_3166_get_country_name_from_alternative(alternative_name: str) -> str:
    this_dir = get_data_dir()
    try:
        data = json.load(open(this_dir + "/iso_countries.json", "r"))
    except Exception as e:
        data = json.load(open(_data_dir_fallback + "/iso_countries.json", "r"))

    for country, alternatives in data.items():
        for alt in alternatives:
            if alt == alternative_name:
                return country
    log.debug(
        f"Alternative_name '{str(alternative_name)}' not found in iso convertion list!"
    )
    return alternative_name


def iso_3166_country_in_iso_format(country: str) -> bool:
    this_dir = get_data_dir()
    try:
        data = json.load(open(this_dir + "/iso_countries.json", "r"))
    except Exception as e:
        data = json.load(open(_data_dir_fallback + "/iso_countries.json", "r"))
    if country in data:
        return True
    return False


class Retrieval(Data):

    """
    The url to the main dataset as csv, if none if supplied the fallback routines get used
    """

    url_csv = ""

    """
    The fallback sources for the downloads can be local/online urls
    or even functions defined in the parent class
    """
    fallbacks = []

    """
    A name mainly for the local file
    """
    name = ""

    def __init__(url, name, url_csv, fallbacks, auto_download=False):
        self.name = name
        self.url_csv = url_csv
        self.fallbacks = fallbacks

        if auto_download:
            self.download_all_available_data()

    def __download_csv_from_source(self, filepath, **kwargs):
        """
        Uses pandas read csv to download the csv file.
        The possible kwargs can be seen in the pandas `documentation <https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html#pandas.read_csv>`_.

        These kwargs can vary for the different parent classes.

        Parameter
        ---------
        filepath : str
            Full path to the desired csv file

        Return
        ------
        :bool
            True if the retrieval was a success, False if it failed
        """
        self.data = pd.read_csv(filepath, **kwargs)
        return True

    def _fallback_handler(self):
        """
        Recursivly iterate over all fallbacks and try to execute subroutines depending on the
        type of fallback.
        """

        def execute_fallback(fallback, i):
            """Helper function to execute the subroutines depending on the type"""
            # Break condition
            success = False
            try:
                # Try to execute the fallback
                if callable(fallback):
                    success = fallback()
                # If it is not executable we try to download from the source
                elif isinstance(fallback, str):
                    success = __download_csv_from_source()
                else:
                    log.info(
                        f"That is weird fallback is not of type string nor a callable function {type(fallback)}"
                    )
                    raise Exception(f"Type error {type(fallback)}")
            except Exception as e:
                info.log(f"Fallback {i+1} failed! {fallback}:{e}")

            # ---------------------------------------------------------------#
            # Break conditions
            # ---------------------------------------------------------------#
            if success:
                log.debug(f"Fallback {i+1} successful! {fallback}")
                return True
            if len(self.fallbacks) == i:
                log.warning(f"ALL fallbacks failed! This should not happen")
                return False

            # ---------------------------------------------------------------#
            # Continue Recursion
            # ---------------------------------------------------------------#
            execute_fallback(self.fallbacks[i + 1], i + 1)

        # Start Recursion
        success = execute_fallback(self.fallbacks[0], 0)
        return success

    def _update_local_files(self, force_local=False) -> bool:
        """
        TODO function that decides if the online files have to be loaded
        """
        if force_local:
            return False
        return True

    def download_all_available_data(self, force_local=False):
        """
        Attempts to download from the main url (self.url_csv) which was given on initialization.
        If this fails download from the fallbacks. It can also be specified to use the local files.

        Parameters
        ----------
        force_local:bool,optional
            If True forces to load the local files.
        """

        def download_helper():
            # First we check if the date of the online file is newer and if we have to download a new file
            # this is done by a function which can be seen above
            try:
                # Try to download from original souce
                self.__download_csv_from_source(self.url_csv)
            except Exception as e:
                # Try all fallbacks
                log.info(f"Failed to download from url {self.url_csv} : {e}")
                self._fallback_handler()
            finally:
                # We save it to the local files
                self.data._save_to_local()

        def local_helper():
            # If we can use a local file we construct the path from the given local name
            try:
                self.data._get_local()
            except Exception as e:
                log.info(f"Failed to load local files! {e} Trying fallbacks!")
                download_helper()

        # ------------------------------------------------------------------------------ #
        # Start of function
        # ------------------------------------------------------------------------------ #

        # If necessary download else use local files
        download_helper() if self._update_local_files(force_local) else local_helper()
