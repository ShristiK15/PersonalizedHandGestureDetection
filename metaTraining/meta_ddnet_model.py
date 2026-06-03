
import torch
import torch.nn as nn

class DDNet(nn.Module):
    def __init__(self, num_classes=25): # num_classes kept for init compatibility, though unused
        super().__init__()

        # Spatial-Temporal Feature Extraction (Exact match to pretraining)
        self.conv1 = nn.Conv1d(63, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Parallel Linear Branches (Exact match to pretraining)
        self.fc_branch1 = nn.Linear(128, 128)
        self.fc_branch2 = nn.Linear(128, 128)

        # NOTE: self.fc_final is intentionally omitted here!

    def forward(self, x):
        x = x.view(x.size(0), 32, 63)
        x = x.permute(0, 2, 1)
        
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        
        x = self.pool(x).squeeze(-1)
        
        out1 = torch.relu(self.fc_branch1(x))
        out2 = torch.relu(self.fc_branch2(x))

        # Return the 256-dimensional feature embedding directly to the ProtoNet
        return torch.cat((out1, out2), dim=1)