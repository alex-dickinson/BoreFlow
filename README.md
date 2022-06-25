# BoreFlow

`BoreFlow` is a Python package for estimating geothermal heat flow from borehole measurements of temperature and thermal conductivity, and from estimates of climatic change. In contrast to most previous approaches to estimating borehole heat flow, it assigns quantitative uncertainties to estimated values. 

We are currently improving `BoreFlow` by writing notebooks to illustrate its use. We hope to submit the code for publication in [Frontiers in Earth Science](https://www.frontiersin.org/journals/earth-science). If you use the software in the meantime, please cite it as:

Dickinson, A., Mather, B., and Ireland, M.T. (_in prep._). BoreFlow: A Python package for estimating terrestrial heat flow from borehole measurements.


## Installation

Base dependencies:

Python 3.8.12

- [`matplotlib >= 3.3.3`](https://matplotlib.org/)
- [`numpy >= 1.19.0`](http://numpy.org)
- [`openpyxl >= 3.0.9`](https://openpyxl.readthedocs.io/en/stable/)
- [`pandas >= 1.0.5`](https://pandas.pydata.org/)
- [`scipy >= 1.5.1`](https://scipy.org)


Extra dependencies for running example notebooks:

- [`jupyter`](https://jupyter-notebook.readthedocs.io/en/stable/)
- [`jupyterlab`](https://jupyterlab.readthedocs.io/en/stable/) 

for running notebooks using either Jupyter Notebook or JupyterLab.


To set up `BoreFlow`, first create conda environment:

```conda create --name NAME```

Activate conda environment:

```conda activate NAME```

Install python 3.8 and jupyter:

```conda install python=3.8 jupyter```

Add conda environment as a Jupyter kernel:

```ipython kernel install --name "NAME" --user```


To install `BoreFlow` with its base dependencies:

Navigate to directory `BoreFlow`



```bash
pip install .
```

To install extra dependencies for running example notebooks:

```bash
pip install .[examples]
```







