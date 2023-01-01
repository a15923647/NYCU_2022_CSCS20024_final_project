# NYCU intro2ML final project
First, join competition on the [kaggle page](https://www.kaggle.com/competitions/tabular-playground-series-aug-2022/data) to gain access to competition data.
Second, fill in fields associated with data in config.py.
# How2train
Setup hyper parameter in config.py.
```shell
$ python3 109550043_Final_train.py
```
Or you may wish to use grid search and kaggle API to submit results.
```shell
$ ./grid_search.sh
```
# How2inference
```shell
$ # python3 109550043_Final_inference.py <path2model>
$ # for pretrained weight
$ python3 109550043_Final_inference.py v1.0_improved_1_train_loss_0.51931.model
```
