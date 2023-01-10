# NYCU intro2ML final project
This is my solution for tabular playground series aug 2022 competition hosted on kaggle.\
It's a combination of data preprocessing, DNN model grid searching and taking average of several models as the final result.

**Table of Contents**

* [NYCU intro2ML final project](#nycu-intro2ml-final-project)
* [Setup environment](#setup-environment)
   * [Platform](#platform)
   * [Install dependencies](#install-dependencies)
   * [Fetch data](#fetch-data)
* [Usage](#usage)
   * [How2train](#how2train)
      * [Setup hyper parameters in config.py.](#setup-hyper-parameters-in-configpy)
      * [Start to train](#start-to-train)
* [How2inference](#how2inference)
* [Data preprocessing](#data-preprocessing)
* [Model architecture](#model-architecture)
* [Experiment Result](#experiment-result)
   * [private score: 0.59472](#private-score-059472)
* [NYCU intro2ML final project](#nycu-intro2ml-final-project)
* [Setup environment](#setup-environment)
   * [Platform](#platform)
   * [Install dependencies](#install-dependencies)
   * [Fetch data](#fetch-data)
* [Usage](#usage)
   * [How2train](#how2train)
      * [Setup hyper parameters in config.py.](#setup-hyper-parameters-in-configpy)
      * [Start to train](#start-to-train)
   * [How2inference](#how2inference)
* [Data preprocessing](#data-preprocessing)
* [Model architecture](#model-architecture)
* [Experiment Result](#experiment-result)
   * [private score: 0.59472](#private-score-059472)

# Setup environment
## Platform
* Device: cpu(Intel I5 12400)
* OS: Ubuntu 22.04
* python: 3.10.6

## Install dependencies
```shell
$ pip3 install -r requirements.txt
```
## Fetch data
First, join competition on the [kaggle page](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data) to gain access to competition data.\
Second, fill in fields associated with data in config.py.

# Usage
## How2train
### Setup hyper parameters in config.py.
```python
TRAIN_CSV_PATH = <path to train.csv>
TEST_CSV_PATH = <path to test.csv>
```
### Start to train
You can train model with hyper parameters set in config.py.
```shell
$ python3 109550043_Final_train.py
```
Or you may wish to use grid search and kaggle API to submit results.
```shell
$ ./grid_search.sh
```
## How2inference
```shell
$ # python3 109550043_Final_inference.py <path2model1> [<path2model2>...]
$ # for pretrained weight
$ python3 109550043_Final_inference.py *.model
```
# Data preprocessing
1. Drop product_code, id, attribute_1
2. Number attribute_0 value
3. Fill nan value with 0 and add a new column to memorize whether this value is a valid value.
4. Apply standard scaler on data to stablize gradient descent.
# Model architecture
You can refer model.py or code segment below.
```python
nn.BatchNorm1d(in_features),
nn.Linear(in_features, 64),
nn.ReLU(),
nn.Linear(64, 32),
nn.ReLU(),
nn.Linear(32, 1),
nn.Sigmoid()
```
# Experiment Result
models link: https://drive.google.com/drive/folders/15ZMkjgpwADFoD_pkejQzDQIvoChkUiSt?usp=sharing
## private score: 0.59472
![best result screenshot](https://github.com/a15923647/NYCU_2022_CSCS20024_final_project/blob/master/result/best.jpg?raw=true)
![private score curve of taking average of several good models](https://github.com/a15923647/NYCU_2022_CSCS20024_final_project/blob/master/result/model1.png?raw=true)
