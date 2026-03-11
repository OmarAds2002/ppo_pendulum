import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.gae import compute_gae, RunningMeanStd


class PPO:
    def __init__(self, agent, cfg):
        self.agent      = agent
        self.cfg        = cfg
        self.optimizer  = optim.Adam(agent.parameters(), lr=cfg.LR, eps=1e-5)
        self.ret_rms    = RunningMeanStd()   # tracks return scale

    def update(self, buffer, next_value):
        advantages, returns = compute_gae(
            buffer.rew_buf,
            buffer.val_buf.detach(),
            buffer.done_buf,
            next_value,
            self.cfg.GAMMA,
            self.cfg.LAMBDA,
            self.cfg.DEVICE,
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize returns for critic — update running stats then divide by std
        self.ret_rms.update(returns)
        returns_norm = self.ret_rms.normalize(returns)

        inds = np.arange(self.cfg.STEPS_PER_ROLLOUT)
        actor_loss = critic_loss = entropy_val = 0.0

        for _ in range(self.cfg.UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, self.cfg.STEPS_PER_ROLLOUT, self.cfg.MINIBATCH_SIZE):
                mb   = inds[start:start + self.cfg.MINIBATCH_SIZE]
                mb_t = torch.tensor(mb).to(self.cfg.DEVICE)

                _, new_logp, entropy, new_value = self.agent.get_action_and_value(
                    buffer.obs_buf[mb_t],
                    buffer.act_buf[mb_t],
                )
                new_value = new_value.squeeze()

                ratio = torch.exp(new_logp - buffer.logp_buf[mb_t])
                surr1 = ratio * advantages[mb_t]
                surr2 = torch.clamp(ratio, 1 - self.cfg.CLIP_EPS, 1 + self.cfg.CLIP_EPS) * advantages[mb_t]

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(new_value, returns_norm[mb_t])
                entropy_val = entropy.mean()

                loss = actor_loss + self.cfg.VF_COEF * critic_loss - self.cfg.ENT_COEF * entropy_val

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.MAX_GRAD_NORM)
                self.optimizer.step()

        return actor_loss.item(), critic_loss.item(), entropy_val.item()