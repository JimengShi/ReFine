# ReFine
This repository contains the source code and datasets for "ReFine: Boosting Time Series Prediction Under Extreme Events by Reweighting and Fine-tuning".

## Repository description
- `data` folder includes data sets used
- `model` folder includes models used
- `training_mlp` folder includes training programs for all datasets. Install the required packages and run cells in the `ipynb` files.
- `postprocess` folder includes the programs for experiment results, visualization, and hyperparameter tuning.
- `saved_models_mlp` folder saves all the trained models at the best checkpoint.
- `saved_models_hyper` folder saves all the trained models while doing hyperparameter tuning for a major hyperparameter `number of frozen layers`.

## Requirements
```bash
conda create -n env_name python=3.8
conda activate env_name
pip3 install -r requirement.txt
