# Digit recognition with PEGASOS

Matteo Cavada.

The report can be found [here](https://github.com/mattyonweb/digit-recognition-MSA/blob/master/report/report.pdf).

## Structure of the repository

- `data` contains the training and test set
- `progetto` contains the runnable code for training and test
  - `main.py` contains the training algorithm and, when invoked, starts the training+execution cycle on the grid of hyperparameters
  - `kfold.py` contains code to perform K-fold operations
  - `test-pretrained.py` executes a test round on one of the pre-trained models found in `results/predictors` 
- `report` contains the `pdf`, `tex` and `org` versions of the report
- `results` stores various results of the performed training+test runs
  - `results.csv` contains raw information on the results of training and test on each single combination of hyperparameters
  - `predictors` contain all the models trained by running `progetto/main.py` in a binary format.

## How to perform training

