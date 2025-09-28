from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x, return_hidden=False):
        x = self.flatten(x)
        h = self.act1(self.fc1(x))
        z = self.act2(self.fc2(h))
        out = self.fc3(z)
        if return_hidden:
            return out, z
        return out
