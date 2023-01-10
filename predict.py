import csv
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import TablarDataset
from model import Net
from config import *

def predict(model_path, store_path='submission.csv'):
    # prepare testing data
    org_df = pd.read_csv(TEST_CSV_PATH)
    test_df = org_df.drop(labels=['id'], axis='columns')
    id_df = org_df['id']
    test_ds = TablarDataset(test_df, x_only=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2*num_gpus, pin_memory=False)
    model = Net(test_ds.in_features)
    # load model
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    out_list = list()
    for X in test_dl:
        X = X.to(device).float()
        preds = model(X).view(-1).data.cpu().numpy()
        # gather a batch of probability and extend to out_list
        out_list.extend(preds.tolist())
    # form output csv data: [output_id, output_probability]
    data = list(zip(id_df, out_list))
    
    # write result to file
    with open(store_path, 'w', newline='') as fin:
        csv_writer = csv.writer(fin)
        csv_writer.writerow(['id', 'failure'])
        csv_writer.writerows(data)
    # also return the result
    return data

if __name__ == '__main__' and len(sys.argv) >= 2:
    predict(sys.argv[1])
