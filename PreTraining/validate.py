import torch

def validate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            _, predicted = outputs.max(1)

            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    acc = 100 * correct / total

    return acc