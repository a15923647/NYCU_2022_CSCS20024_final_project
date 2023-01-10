import time
import torch
from tqdm import tqdm
from config import *

def train(net, train_dataloader, valid_dataloader, criterion, optimizer, scheduler=None, epochs=10, device='cpu', checkpoint_epochs=10, val_interval=1):
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    start = time.time()
    last_store_path = ''
    print(f'Training for {epochs} epochs on {device}')

    for epoch in tqdm(range(1, epochs+1)):
        print(f"Epoch {epoch}/{epochs}")
        net.train()
        train_loss = torch.tensor(0., device=device)
        train_accuracy = torch.tensor(0., device=device)
        pred_ones = torch.tensor(0., device=device)
        for X, y in train_dataloader:
            X = X.to(device)
            y = y.to(device)
            y = y.view(-1, 1)
            preds = net(X)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_loss += loss * train_dataloader.batch_size
                train_accuracy += (preds.detach().round() == y.detach()).sum()
                pred_ones += (preds.detach().round() == 1).sum()
        print(f'Training loss: {train_loss/len(train_dataloader.dataset):.5f}')
        print(f'Training accuracy: {100*train_accuracy/len(train_dataloader.dataset):.5f}')
        print(f'Training ones: {100*pred_ones/len(train_dataloader.dataset):.5f}')
        
        if valid_dataloader is not None and epoch % val_interval == 0:
            net.eval()
            valid_loss = torch.tensor(0., device=device)
            valid_accuracy = torch.tensor(0., device=device)
            pred_ones = torch.tensor(0., device=device)
            with torch.no_grad():
                for X, y in valid_dataloader:
                    X = X.to(device)
                    y = y.to(device)
                    y = y.view(-1, 1)
                    preds = net(X)
                    loss = criterion(preds, y)
                    
                    valid_loss += loss * valid_dataloader.batch_size
                    valid_accuracy += (preds.detach().round() == y).sum()
                    pred_ones += (preds.detach().round() == 1).sum()
            print(f'Valid loss: {valid_loss/len(valid_dataloader.dataset):.5f}')
            print(f'Valid accuracy: {100*valid_accuracy/len(valid_dataloader.dataset):.5f}')
            print(f'Valid ones: {100*pred_ones/len(valid_dataloader.dataset):.5f}')

            # store the one with minimal validation loss
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                s = f"loss_{valid_loss/len(valid_dataloader.dataset):.5f}"
                last_store_path = f"./v{PRIMARY_VERSION}.{MINOR_VERSION}_improved_{epoch}_{s}.model"
                torch.save(net.state_dict(), last_store_path)

        elif valid_dataloader is None:
            # if no validation dataset, by default store the one with minimal train loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                s = f'train_loss_{train_loss/len(train_dataloader.dataset):.5f}'
                last_store_path = f"./v{PRIMARY_VERSION}.{MINOR_VERSION}_improved_{epoch}_{s}.model"
                torch.save(net.state_dict(), last_store_path)
            elif STORE_TRAIN_ALL:
                # or wish to store all models to prevent overfiting on training data
                s = f'train_loss_{train_loss/len(train_dataloader.dataset):.5f}'
                last_store_path = f"./v{PRIMARY_VERSION}.{MINOR_VERSION}_improved_{epoch}_{s}.model"
                torch.save(net.state_dict(), last_store_path)
            
        if scheduler is not None:
            scheduler.step()
        # store model and optimizer state in every checkpoint_epochs
        if epoch % checkpoint_epochs == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'./34checkpoint{epoch}.pth.tar')
            torch.save(net.state_dict(), f"./v{PRIMARY_VERSION}.{MINOR_VERSION}_{epoch}.model")
        print()

    end = time.time()
    print(f'Total training time: {end-start:.1f} seconds')
    return net, last_store_path
