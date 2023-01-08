# NYCU intro2ML final project
# Setup environment
## Install dependencies
```shell
$ pip3 install -r requirements.txt
```
## Fetch data
First, join competition on the [kaggle page](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data) to gain access to competition data.
Second, fill in fields associated with data in config.py.

# How2train
## Setup hyper parameters in config.py.
## Start to train
```shell
$ python3 109550043_Final_train.py
```
Or you may wish to use grid search and kaggle API to submit results.
```shell
$ ./grid_search.sh
```
# How2inference
```shell
$ # python3 109550043_Final_inference.py <path2model1> [<path2model2>...]
$ # for pretrained weight
$ python3 109550043_Final_inference.py *.model
```
# Result
models link: https://drive.google.com/drive/folders/15ZMkjgpwADFoD_pkejQzDQIvoChkUiSt?usp=sharing
## private score: 0.59421
