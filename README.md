[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kchen8921/DA-HEF/master)

# Data Assimilation 

[![N|Solid](https://upload.wikimedia.org/wikipedia/en/thumb/1/17/Pacific_Northwest_National_Laboratory_logo.svg/200px-Pacific_Northwest_National_Laboratory_logo.svg.png)](https://www.pnnl.gov/)

In the project, data assimilation methods such as EnKF and ES-MDA are used to estimate hydrologic exchange flux between surface water and groundwater by assimilating observed field data at Hanford site (e.g. temperature, hydaulic heads...). This repository provides the entire workflow for the implementation of DA. Objectives include but not limited to:

  - Estimate permeability and thermal conductivity in the riverbed
  - Infer subdaily hydrologic exchange flux

# Contents




# Installation and Configuration

1. The workflow is written in [Jupyter notebook](http://jupyter.org/) which supports both Python and R. A recommended distribution of Jupyter notebook is [Anaconda](https://www.anaconda.com/download/).
  (1) To start Jupyter notebook on Mac/Linux after installation of Anaconda, typing the following command in terminal:
    ```sh
    jupyter notebook
    ```
    (2) To start Jupyter notebook on Windows, just click the desktop icon. 
2. The numerical model for subsurface flow at Hanford site is built using [PFLOTRAN](http://www.pflotran.org/), a massively parallel reactive flow and transport model for describing surface and subsurface processes.
3. The workflow is adapted to the supercomputers at [National Energy Research Scientific Computing Center (NERSC)](http://www.nersc.gov/).



