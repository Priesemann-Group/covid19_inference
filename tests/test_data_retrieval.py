# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-05-03 14:40:00
# @Last Modified: 2020-07-07 12:06:02
# ------------------------------------------------------------------------------ #
# failry rudimentary. todo:
#   * go offline and check retrieval of local stuff
#   * check meaningful output. (should be doable for the cached stuff)
# ------------------------------------------------------------------------------ #
import datetime


def test_google():
    import covid19_inference as cov

    gl = cov.data_retrieval.GOOGLE(False)
    gl.download_all_available_data(force_download=True)

    # Force load offline data
    gl.download_all_available_data(force_local=True)
    # automatic detection
    gl.download_all_available_data()

    # Test different filter function
    gl.get_changes(
        country="Germany",
        data_begin=datetime.datetime(2020, 3, 15),
        data_end=datetime.datetime(2020, 3, 25),
    )


def test_rki():
    import covid19_inference as cov

    rki = cov.data_retrieval.RKI(False)
    rki.download_all_available_data(force_download=True)

    # Force load offline data
    rki.download_all_available_data(force_local=True)
    rki.download_all_available_data()

    # Test different filter function
    rki.get_total("deaths", "Sachsen")
    rki.get_new(
        "confirmed",
        "Sachsen",
        data_begin=datetime.datetime(2020, 3, 15),
        data_end=datetime.datetime(2020, 3, 25),
    )


def test_jhu():
    import covid19_inference as cov

    jhu = cov.data_retrieval.JHU(False)
    jhu.download_all_available_data(force_download=True)

    # Force load offline data
    jhu.download_all_available_data(force_local=True)
    jhu.download_all_available_data()

    jhu.get_total(
        "confirmed",
        country="Italy",
        data_begin=datetime.datetime(2020, 3, 15),
        data_end=datetime.datetime(2020, 3, 25),
    )
    jhu.get_new(
        "confirmed",
        country="Italy",
        data_begin=datetime.datetime(2020, 3, 15),
        data_end=datetime.datetime(2020, 3, 25),
    )


def test_owd():
    import covid19_inference as cov

    owd = cov.data_retrieval.OWD(False)
    owd.download_all_available_data(force_download=True)

    # Force load offline data
    owd.download_all_available_data(force_local=True)
    owd.download_all_available_data()

    owd.get_total(
        "confirmed",
        country="Italy",
        data_begin=datetime.datetime(2020, 3, 15),
        data_end=datetime.datetime(2020, 3, 25),
    )
    owd.get_new(
        "tests",
        country="Belgium",
        data_begin=datetime.datetime(2020, 3, 15),
        data_end=datetime.datetime(2020, 3, 25),
    )


def test_ft():
    import covid19_inference as cov

    ft = cov.data_retrieval.FINANCIAL_TIMES(False)
    ft.download_all_available_data(force_download=True)

    # Force load offline data
    ft.download_all_available_data(force_local=True)
    ft.download_all_available_data()

    ft.get(
        "excess_deaths",
        country="Italy",
        data_begin=datetime.datetime(2020, 3, 15),
        data_end=datetime.datetime(2020, 3, 25),
    )


def test_oxcgrt():
    import covid19_inference as cov

    gov_pol = cov.data_retrieval.OxCGRT(False)
    gov_pol.download_all_available_data(force_download=True)

    # Force load offline data
    gov_pol.download_all_available_data(force_local=True)
    gov_pol.download_all_available_data()


def test_epistat():
    import covid19_inference as cov

    epi = cov.data_retrieval.countries.Belgium(False)
    epi.download_all_available_data(force_download=True)

    # Force load offline data
    epi.download_all_available_data(force_local=True)
    epi.download_all_available_data()

    epi.get_new()

    epi.get_total()
