# Simulating predictions of classification models - Code for experiments
This repository contains the code to reproduce the computational experiments reported in the paper: ___Simulating predictions of classification models to evaluate predict-then-optimize methods___.

## Overview

The experiments are implemented in two Jupyter notebooks:
- `binary_experiments.ipynb`: Reproduces the results for binary classification settings.
- `multiclass_experiments.ipynb`: Reproduces the results for multiclass classification settings.

These notebooks generate the detailed computational results plotted in Figures 1–3 in the paper. 
Results are printed directly to the console for further processing.

Additionally, the file `siml.py` defines the core functions used to simulate predictions from classification models, which are used throughout both notebooks.

## Requirements
To run the code, you will need the following Python packages:
- scikit-learn
- xgboost
- pandas
- numpy

## Citation
If you use this code in your research, please cite the paper accordingly.