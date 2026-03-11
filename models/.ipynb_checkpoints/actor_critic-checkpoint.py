import torch.nn as nn
from models.actor  import Actor
from models.critic import Critic


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size):
        super().__init__()
        self.actor  = Actor(obs_dim, act_dim, hidden_size)
        self.critic = Critic(obs_dim, hidden_size)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action, log_prob, entropy = self.actor(x, action)
        value = self.critic(x)
        return action, log_prob, entropy, value
