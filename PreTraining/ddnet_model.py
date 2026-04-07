import torch
import torch.nn as nn

class DDNet(nn.Module):

    def __init__(self,num_classes=25):

        super().__init__()

        self.conv1 = nn.Conv1d(63,128,3,padding=1)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128,128,3,padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # self.fc1 = nn.Linear(128,128)
        # self.fc2 = nn.Linear(128,num_classes)

        self.fc_branch1 = nn.Linear(128, 128)
        self.fc_branch2 = nn.Linear(128, 128)
        self.fc_final = nn.Linear(256, num_classes)

    def forward(self,x):

        x = x.view(x.size(0),32,63)
        x = x.permute(0,2,1)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        x = self.pool(x).squeeze(-1)

        # x = torch.relu(self.fc1(x))

        # return self.fc2(x)

        # Pass features through parallel branches
        out1 = torch.relu(self.fc_branch1(x))
        out2 = torch.relu(self.fc_branch2(x))

        # Concatenate branches
        concat_features = torch.cat((out1, out2), dim=1)

        # Classification output
        return self.fc_final(concat_features)