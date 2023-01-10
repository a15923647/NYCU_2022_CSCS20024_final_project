import torch
PRIMARY_VERSION = 1
MINOR_VERSION = 0
TRAIN_CSV_PATH = "train.csv"
TEST_CSV_PATH = "test.csv"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()
checkpoint_epochs = 100
BATCH_SIZE = 64
lr = 5e-4
weight_decay = 3e-4
betas = (0.9, 0.999)
epochs = 30
full_epochs = 15
val_interval = 1
CRITERION = lambda: torch.nn.BCELoss()
CHECKPOINT_PATH = ''
STORE_TRAIN_ALL = True
