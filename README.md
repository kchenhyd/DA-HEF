[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kchen8921/DA-HEF/master)

# Data Assimilation 

[![N|Solid](https://upload.wikimedia.org/wikipedia/en/thumb/1/17/Pacific_Northwest_National_Laboratory_logo.svg/200px-Pacific_Northwest_National_Laboratory_logo.svg.png)](https://www.pnnl.gov/)

In the project, ensemble smoother-multiple data assimilation (ES-MDA) based data assimilation approach is used to estimate hydrologic exchange flux between surface water and groundwater by assimilating observed field data at Hanford site (e.g. temperature, hydaulic heads...). This repository provides the entire workflow for the implementation of DA. Objectives include but not limited to:

  - Estimate permeability and thermal conductivity in the riverbed
  - Infer subdaily hydrologic exchange flux



# Installation and Configuration

1. The numerical model for subsurface flow at Hanford site is built using [PFLOTRAN](http://www.pflotran.org/), a massively parallel reactive flow and transport model for describing surface and subsurface processes.
2. The workflow is adapted to the supercomputers at [National Energy Research Scientific Computing Center (NERSC)](http://www.nersc.gov/).



