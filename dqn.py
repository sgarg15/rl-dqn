import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # The net is simply state_dim -> 128 -> 128 -> action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            #Activation function to introduce non-linearity (for positive values it returns the input, for negative values it returns 0)
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)