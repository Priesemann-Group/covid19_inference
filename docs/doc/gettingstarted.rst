Getting Started
===============

.. automodule:: covid19_inference

Installation
------------

There exist three different possibilities for usage:

1. Clone the repository, with the latest release:

.. code-block:: console

    git clone https://github.com/Priesemann-Group/covid19_inference


2. Install the module via pip

.. code-block:: console

    pip install covid19_inference


3. Run the notebooks directly in Google Colab. At the top of the notebooks files
there should be a symbol which opens them directly in a Google Colab instance.

First Steps
-----------

To get started, we recommend to look at one of the currently two example notebooks:

1. `SIR model with one german state <https://github.com/Priesemann-Group/covid19_inference/blob/master/scripts/example_one_bundesland.ipynb>`_
    This model is similar to the one discussed in our paper: `Inferring COVID-19 spreading rates and potential change points for case number forecasts <https://arxiv.org/abs/2004.01105>`_.
    The difference is that the delay between infection and report is now lognormal distributed and not
    fixed.

2. `Hierarchical model of the German states  <https://github.com/Priesemann-Group/covid19_inference/blob/master/scripts/example_bundeslaender.ipynb>`_
    This builds a hierarchical Bayesian model of the states of Germany. Caution, seems to be currently broken!

We can for example recommend the following articles about Bayesian modeling:

As a introduction to Bayesian statistics and the python package (PyMc) that we use:
https://docs.pymc.io/notebooks/api_quickstart.html

This is a good post about hierarchical Bayesian models in general:
https://statmodeling.stat.columbia.edu/2014/01/21/everything-need-know-bayesian-statistics-learned-eight-schools/
