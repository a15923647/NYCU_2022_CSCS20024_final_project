# NYCU intro2ML final project
This is my solution for tabular playground series aug 2022 competition hosted on kaggle.\
It's a combination of data preprocessing, DNN model grid searching and taking average of several models as the final result.\

# Setup environment
## Platform
* Device: cpu(Intel I5 12400)
* OS: Ubuntu 22.04

## Install dependencies
```shell
$ pip3 install -r requirements.txt
```
## Fetch data
First, join competition on the [kaggle page](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data) to gain access to competition data.\
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
## private score: 0.59472
![best result screenshot](https://github.com/a15923647/NYCU_2022_CSCS20024_final_project/blob/master/result/best.jpg?raw=true)
![private score curve of taking average of several good models](https://github.com/a15923647/NYCU_2022_CSCS20024_final_project/blob/master/result/model1.png?raw=true)
