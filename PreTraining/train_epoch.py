import torch

def train_epoch(model, loader, optimizer, criterion, device):

    model.train()

    correct = 0
    total = 0
    total_loss = 0

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        _, predicted = outputs.max(1)

        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    acc = 100 * correct / total

    return total_loss / len(loader), acc