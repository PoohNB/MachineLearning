# ml-flow-regression

This repository showcases a practical implementation of linear regression using numpy and mlflow, complemented with a Flask API for deployment. The experiment employs various regression algorithms including polynomial, ridge, lasso, elastic net, and normal. The accompanying notebook allows for the selection of initial weight methods (zero or xavier) and the application of momentum. 

## project structure

- `datasets`: Holds the dataset used for training and testing the model.
- `Regularization-mlflow.ipynb`: Jupyter notebooks used for data preprocessing and model training.
- `mlartifacts`: mlflow models
- `mlruns`: mlflow logs
- `models`: Stores the scaler.  
- `ML-website`: contains flask website
- 
## Installation

1. clone repository

```
git clone https://github.com/iforgeti/ml-flow-regression.git
```

2. Navigate to the project directory

```
cd ml-flow-regression
cd ML-website
```

3. run docker
   
```
docker compose up -d
```