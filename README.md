# HD2AT

This is a PyTorch implementation of the paper: Heterogeneous Dual-Dynamic Attention Network for Modeling Mutual Interplay of Stocks.

## Requirements
* python==3.7.13
* torch==1.8.0
* sklearn==1.0.2
* talib==0.4.19
* numpy==1.21.6
* pandas==1.3.5

## How to train the model
1. Run `estimators.ipynb`.
This script would run the preprocessing for raw data and construct the intervention graphs.
2. Run `main.ipynb`.
This script would build an HD2AT model, and then train and evaluate the model.
