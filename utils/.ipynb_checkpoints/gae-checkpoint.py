import torch
import numpy as np


def compute_gae(rew_buf, val_buf, done_buf, next_value, gamma, lam, device):
    """
    rew_buf   : np.ndarray [T]
    val_buf   : torch.Tensor [T]  (detached)
    done_buf  : np.ndarray [T]
    next_value: float — V(s_{T+1}), 0.0 if last step was terminal
    """
    T = len(rew_buf)
    advantages = torch.zeros(T, device=device)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(done_buf[t])
        next_val = val_buf[t + 1].item() if t < T - 1 else next_value
        delta = float(rew_buf[t]) + gamma * next_val * next_non_terminal - val_buf[t].item()
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = float(last_gae)

    returns = advantages + val_buf
    return advantages, returns


class RunningMeanStd:
    """Welford online algorithm for running mean and variance."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        batch_mean = x.mean().item()
        batch_var = x.var().item()
        batch_count = x.shape[0]
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.var = (
            self.var    * self.count
            + batch_var * batch_count
            + delta**2  * self.count * batch_count / total
        ) / total
        self.count  = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Divide by running std only — preserves sign of values."""
        return x / (self.var ** 0.5 + 1e-8)