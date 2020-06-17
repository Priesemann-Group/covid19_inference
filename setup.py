from setuptools import setup, find_namespace_packages
import re

# read the contents of your README file
from os import path

with open("README.md") as f:
    long_description = f.read()

verstr = "unknown"
try:
    verstrline = open("covid19_inference/_version.py", "rt").read()
except EnvironmentError:
    pass
else:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, verstrline, re.M)
    if mo:
        verstr = mo.group(1)
    else:
        raise RuntimeError("unable to find version in covid19_inference/_version.py")

setup(
    name="covid19_inference",
    author="Jonas Dehning, Johannes Zierenberg, F. Paul Spitzner, Michael Wibral, Joao Pinheiro Neto, Michael Wilczek, Viola Priesemann",
    author_email="jonas.dehning@ds.mpg.de",
    packages=find_namespace_packages(),
    url="https://github.com/Priesemann-Group/covid19_inference",
    description="Toolbox for inference and forecast of the spread of the Coronavirus.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6.0",
    version=verstr,
    install_requires=["pymc3", "matplotlib", "numpy", "pandas", "theano",],
)
