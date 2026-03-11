import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        # FIX 4: Orthogonal init — PPO standard, prevents early loss spikes
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.constant_(layer.bias, 0.0)
        # Output layer gets smaller gain so initial values are near 0
        nn.init.orthogonal_(self.network[-1].weight, gain=1.0)
        nn.init.constant_(self.network[-1].bias, 0.0)

    def forward(self, state):
        return self.network(state)