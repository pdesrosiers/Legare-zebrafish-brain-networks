# Structural and genetic determinants of zebrafish functional brain networks

Code associated with the "Structural and genetic determinants of zebrafish functional brain networks" paper, currently available on bioRxiv.

# Overview

Data analyses are conducted in multiple Jupyter notebooks that are mostly split into 2 categories: Some that compute results for specific figures of the paper (`FigureX-Analysis` notebooks), and some that take these results to generate figures (`FigureX-Layout` notebooks) for subsequent modifications using Inkscape. Notebooks import functions in adjacent `.py` files and perform analyses on a processed calcium imaging dataset that can be downloaded elsewhere (roughly 20 GB of data). If notebooks are executed in the correct order (see below), most of the figure panels can be reproduced, with the exception of a few figures that require raw data (over a terabyte, available upon reasonable request). Supplementary analyses are done in `Supp-` notebooks.

# Detailed list

- `Figure1-Analysis` : This notebook briefly inspects the calcium imaging dataset, counts neurons in every brain region, excludes regions with few cells, detrends fluorescence time series, computes functional networks, and performs the inter-individual similarity/fingerprinting analysis; statistical validations of fingerprinting are conducted in a separate notebook.
- `Figure1-Layout`: Loads previously calculated results and generates **Figure 1**.
- `Figure2-Layout`: Loads previously calculated results and generates **Figure 2**.
- `Figure3-Layout`: Loads previously calculated results and generates **Figure 3**.
- `Figure4-Layout`: Loads previously calculated results and generates **Figure 4**.
- `Figure5-Layout`: Loads previously calculated results and generates **Figure 5**.
- `Figure6-Layout`: Loads previously calculated results and generates **Figure 6**.
- `Figure7-Layout`: Loads previously calculated results and generates **Figure 7**.
- `Supp-Figures` : Generates **Supplementary Figures S1, S2, S5, S8, S13, S14, S15, S16, S20, S21, S22, S23, S25**.
- `Supp-Fingerprinting` : Compares different methods to fingerprint the identity of larvae from calcium imaging data, then generates **Supplementary Figure S6**.
- `Supp-SC-Validations` : Validation of the boundary expansion procedure for SC calculation, related to **Supplementary Figure S9**.
- `Supp-SCCM` : Defines the connectivity null model, generates null connectivity matrices from SC matrices generated beforehand, and generates **Supplementary Figures S10, S11, & S12**.
- `Supp-SparseSC` : Replication of key results using the sparse SC matrix, related to **Supplementary Figure S18**.
- `Supp-Gradients` : Computes FC gradients in various FC definitions, then generates **Supplementary Figure S24**.

# Authors

- Antoine Légaré (antoine.legare.1@ulaval.ca)
- Patrick Desrosiers (patrick.desrosiers@phy.ulaval.ca)

For any questions regarding the repository, please contact us!

