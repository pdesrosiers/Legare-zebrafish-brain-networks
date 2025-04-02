# Structural and genetic determinants of zebrafish functional brain networks

Code associated with the "Structural and genetic determinants of zebrafish functional brain networks" paper, currently available on bioRxiv.

# Overview

Data analyses are conducted in separate Jupyter notebooks that are mostly split into 2 categories: Some that compute results for specific figures of the paper (`-Analysis` notebooks), and some that take these results to generate figures (`-Layout` notebooks) for subsequent modifications using Inkscape. Notebooks import functions in adjacent `.py` files and perform analyses on a processed calcium imaging dataset that can be downloaded elsewhere (roughly 20 GB of data). If notebooks are executed in the correct order (see below), most of the figures panels can be reproduced, with the exception of a few figures that require raw data (over a terabyte, available upon reasonable request). Supplementary analyses are done in `Supp-` notebooks.

# Detailed list

- `Figure1-Analysis` : This notebook briefly inspects the calcium imaging dataset, counts neurons in every brain region, excludes regions with few cells, detrends fluorescence time series, computes functional networks, and performs the inter-individual similarity/fingerprinting analysis.

# Authors

For any questions regarding the repository, please contact us:

- Antoine Légaré (antoine.legare.1@ulaval.ca)
- Patrick Desrosiers (patrick.desrosiers@phy.ulaval.ca)
