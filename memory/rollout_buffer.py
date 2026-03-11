import torch
import numpy as np


class RolloutBuffer:
    def __init__(self, steps, obs_dim, act_dim, device):
        self.steps   = steps
        self.device  = device
        self.obs_buf  = torch.zeros(steps, obs_dim).to(device)
        self.act_buf  = torch.zeros(steps, act_dim).to(device)
        self.logp_buf = torch.zeros(steps).to(device)
        self.val_buf  = torch.zeros(steps).to(device)
        self.rew_buf  = np.zeros(steps, dtype=np.float32)
        self.done_buf = np.zeros(steps, dtype=np.float32)
        self.step     = 0

    def store(self, obs, action, log_prob, value, reward, done):
        self.obs_buf[self.step]  = obs
        self.act_buf[self.step]  = action
        self.logp_buf[self.step] = log_prob
        self.val_buf[self.step]  = value
        self.rew_buf[self.step]  = reward
        self.done_buf[self.step] = float(done)
        self.step += 1

    def clear(self):
        self.step = 0
