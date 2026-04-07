import sys
import os

# Get the path of the current script's directory, then go one level up
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)


import torch
import torch.nn as nn
import torch.optim as optim

from ddnet_model import DDNet
from dataloader_setup import get_dataloaders
from train_epoch import train_epoch
from validate import validate
from config import *


def main():

    train_loader, val_loader = get_dataloaders()

    model = DDNet(num_classes=25).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_val_acc = 0

    for epoch in range(epochs):

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_acc = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "weights/ddnet_pretrained.pth"
            )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()