# Légaré et al. 2024

Code for the "Structural and genetic constraints on zebrafish brain networks". Work in progress.

# Overview

Notebooks are mostly split into 2 categories: Some that compute results for specific figures of the paper (`-Analysis` notebooks), and some that take these results to export figures (`-Layout` notebooks) for subsequent modification using Inkscape. Notebooks import functions in adjacent `.py` files and perform analyses on a processed calcium imaging dataset that can be downloaded elsewhere from X (roughly 20 GB of data). If notebooks are executed in the correct order (see below), most of the figure panels can be reproduced, with the exception of a few figures that require raw data (over a terabyte, available upon reasonable request).

# Detailed list

- `Figure1-Analysis` : This notebook briefly inspects the calcium imaging dataset, counts neurons in every brain region, excludes regions with few cells, detrends fluorescence time series, computes functional networks, and performs the inter-individual similarity analysis.


# Authors

Antoine Légaré (antoine.legare.1@ulaval.ca)
Patrick Desrosiers (patrick.desrosiers@phy.ulaval.ca)
