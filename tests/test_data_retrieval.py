# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-05-03 14:40:00
# @Last Modified: 2020-05-03 16:37:14
# ------------------------------------------------------------------------------ #
# failry rudimentary. todo:
#   * go offline and check retrieval of local stuff
#   * check meaningful output. (should be doable for the cached stuff)
# ------------------------------------------------------------------------------ #


def test_google():
    import covid19_inference as cov

    gl = cov.data_retrieval.GOOGLE(False)
    gl.download_all_available_data()


def test_rki():
    import covid19_inference as cov

    rki = cov.data_retrieval.RKI(False)
    rki.download_all_available_data()


def test_jhu():
    import covid19_inference as cov

    jhu = cov.data_retrieval.JHU(False)
    jhu.download_all_available_data()
