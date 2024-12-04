import torch
import torch.nn as nn
from torch.distributions import Normal

# Policy network for action distribution
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        # Create hidden layers
        for dim_in, dim_out in zip([input_dim] + hidden_dims[:-1], hidden_dims):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
        # Final layer for mean of action distribution
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.mean_network = nn.Sequential(*layers)
        self.log_std = nn.Parameter(-0.5 * torch.ones(output_dim))

    def forward(self, state):
        mean = self.mean_network(state)
        std = torch.exp(self.log_std)
        return mean, std

    def sample_action(self, state):
        mean, std = self(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.detach().numpy(), log_prob.detach()

# Value network for predicting state value
class CriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        # Create hidden layers
        for dim_in, dim_out in zip([input_dim] + hidden_dims[:-1], hidden_dims):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
        # Output single scalar
        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.value_network = nn.Sequential(*layers)

    def forward(self, state):
        return self.value_network(state).squeeze(-1)
