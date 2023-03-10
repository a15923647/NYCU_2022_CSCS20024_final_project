import torch
import pandas as pd
from config import *
from model import Net
from predict import predict
from train import train
from dataset import TablarDataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

# Random Over Sampling
org_all_train_df = pd.read_csv(TRAIN_CSV_PATH).drop(labels=['id'], axis='columns')
zeros, ones = org_all_train_df.failure.value_counts()
if zeros > ones:
    zeors_df = org_all_train_df[org_all_train_df['failure'] == 0]
    ones_df = org_all_train_df[org_all_train_df['failure'] == 1].sample(zeros, replace=True)
else:
    zeors_df = org_all_train_df[org_all_train_df['failure'] == 1].sample(zeros, replace=True)
    ones_df = org_all_train_df[org_all_train_df['failure'] == 0]
all_train_df = pd.concat([zeors_df, ones_df], axis=0)
# mix two class records together
all_train_df = shuffle(all_train_df)

# train samples : validation samples = 7 : 3
train_df_len = int(len(all_train_df) * 0.7)
train_df = all_train_df.iloc[:train_df_len]
val_df = all_train_df.iloc[train_df_len:]
train_ds = TablarDataset(train_df, x_only=False)
val_ds = TablarDataset(val_df, x_only=False)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2*num_gpus, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2*num_gpus, pin_memory=True)

model = Net(train_ds.in_features)
model = model.to(device)
if CHECKPOINT_PATH:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device(device)))
##################################### train ###################################
criterion = CRITERION()
criterion.to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
model, last_store_path = train(model, train_dl, val_dl, criterion, optimizer, None, epochs, device, checkpoint_epochs=checkpoint_epochs, val_interval=val_interval)
print(last_store_path)
#################################### train with full data #####################
full_ds = TablarDataset(all_train_df, x_only=False)
full_dl = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2*num_gpus, pin_memory=True)
criterion = CRITERION()
criterion.to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
model.load_state_dict(torch.load(last_store_path, map_location=torch.device(device)))
model, last_store_path = train(model, full_dl, None, criterion, optimizer, None, full_epochs, device, checkpoint_epochs=checkpoint_epochs, val_interval=val_interval)
print(last_store_path)
################################### predict ##################################
predict(last_store_path)
