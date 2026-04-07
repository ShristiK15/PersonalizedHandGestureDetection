import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epochs = 50
learning_rate = 0.001
weight_decay = 0.001
batch_size = 128