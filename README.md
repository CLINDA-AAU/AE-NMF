# The AE-NMF Repository

This repository contains scripts for extracting and analyzing signature- and exposure matrices using Non-negative Matrix Factorization (NMF), convex NMF (C-NMF), and Autoencoder-NMF (AE-NMF) methods. The scripts in this repository are the methods used in the paper 'On the relation between auto-encoders and non-negative matrix factorization, and their application for mutational signature extraction' by Egendal et al. 

Genomic data used in these analyses are the subsets of the Genomics England 100 000 Genomes initiative acessed on [Zendo](https://zenodo.org/records/5571551), and COSMIC validation were performed with the COSMIC v3.4 SBS mutational signatures acessed on the  [COSMIC Website](https://cancer.sanger.ac.uk/signatures/downloads/).

## Overview
This repository contains scripts to:
* Perform de novo mutational signature extraction using NMF (`NMF.py`), convex NMF (`CNMF.py`) and autoencoders with different non-negativity constraints (`AENMF.py`).
* Apply these on simulated data and cancer genomics data (`generate_results.py`, `simulate_example.py`)
* Analyze and validate the resulting factorizations (`calculate_consistency.py`, `COSMIC_validation.r`) 

Moreover, all the factor matrices used in the paper above are given in the folder `generated_data`

## Authors

* **Ida Egendal** - [idabue](https://github.com/idabue)

## Acknowledgements
* **Rasmus Froberg Brøndum**
* **Marta Pelizzola**
* **Asger Hobolth**
* **Martin Bøgsted**
