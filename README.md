# "You Can't Fix What You Can't Measure": Privately Measuring Demographic Performance Disparities in Federated Learning

This repository contains the source code and the data to reproduce and validate the results of the paper *"You Can't Fix What You Can't Measure": Privately Measuring Demographic Performance Disparities in Federated Learning*, submitted to the 16th volume of the Proceedings of VLDB Endowment (2023).

## Organization of this repository
The following is the description of the directory structure of this repository:

- `src`: contains the Python implementation of the mechanisms described in the paper.
- `notebooks/validation`: contains the Jupyter notebooks that validate the theoretical results regarding the MSE (i.e., variance) of the mechanism estimators.
- `notebooks/empirical_evaluation`: contains the Jupyter notebooks that implement the experiments for the empirical evaluation (Section 5.5). In order to run these notebooks, you will need to unzip the activity detection dataset (`data.zip`).
