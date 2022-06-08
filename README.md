# BoreFlow


Magnetic data is one of the most common geophysics datasets available on the surface of the Earth. Curie depth is the depth at which rocks lose their magnetism. The most prevalent magnetic mineral is magnetite, which has a Curie point of 580°C, thus the Curie depth is often interpreted as the 580°C isotherm.

Current methods to derive Curie depth first compute the (fast) Fourier transform over a square window of a magnetic anomaly that has been reduced to the pole. The depth and thickness of magnetic sources is estimated from the slope of the radial power spectrum. `pycurious` implements the Tanaka *et al.* (1999) and Bouligand *et al.* (2009) methods for computing the thickness of a buried magnetic source. `pycurious` ingests maps of the magnetic anomaly and distributes the computation of Curie depth across multiple CPUs. Common computational workflows and geospatial manipulation of magnetic data are covered in the Jupyter notebooks bundled with this package.



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



#### Citation

`BoreFlow` is in preparation for submission to [Frontiers in Earth Science](https://www.frontiersin.org/journals/earth-science). If you use the software, please cite it as:

Dickinson, A., Mather, B., and Ireland, M.T. (_in prep._). BoreFlow: A Python module for estimating geothermal heat flow from borehole measurements.



