Getting Started
===============

.. automodule:: covid19_inference

Installation
------------

There exists three different possiblities to run the models:

1. Clone the repository:

.. code-block:: console

    git clone https://github.com/Priesemann-Group/covid19_inference


2. Install the module via pip

.. code-block:: console

    pip install git+https://github.com/Priesemann-Group/covid19_inference.git


3. Run the notebooks directly in Google Colab. At the top of the notebooks files
there should be a symbol which opens them directly in a Google Colab instance.

First Steps
-----------

To get started, we recommend to look at one of the currently two example notebooks:

1. `SIR model with changes points <https://github.com/Priesemann-Group/covid19_inference_forecast/blob/master/scripts/example_script_covid19_inference.ipynb>`_
    This model is extensively discussed in our paper: `Inferring COVID-19 spreading rates and potential change points for case number forecasts <https://arxiv.org/abs/2004.01105>`_

2. `Hierarchical model of the German states  <https://github.com/Priesemann-Group/covid19_inference/blob/master/scripts/example_bundeslaender.ipynb>`_
    This builds a hierarchical bayesian model of the states of Germany
