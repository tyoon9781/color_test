import torch.nn as nn
import torch.nn.functional as F

class ColorNetV1(nn.Module):
    def __init__(self):
        super(ColorNetV1, self).__init__()
        self.fc1 = nn.Linear(3, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
