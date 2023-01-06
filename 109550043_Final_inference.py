import csv
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TablarDataset
from model import Net
from config import *
from merge_res import avg
from predict import predict

if __name__ == '__main__':
    if len(sys.argv) == 2:
        predict(sys.argv[1])
    elif len(sys.argv) != 1:
        avg(sys.argv[1:])
        
