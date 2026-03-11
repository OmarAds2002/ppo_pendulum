import numpy as np
import torch
from config import Config
from env.make_env import make_env
from models.actor_critic import ActorCritic
from memory.rollout_buffer import RolloutBuffer
from algorithms.ppo import PPO


def train():
    cfg = Config()
    env = make_env(cfg.ENV_NAME)

    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    act_low  = env.action_space.low
    act_high = env.action_space.high

    agent  = ActorCritic(obs_dim, act_dim, cfg.HIDDEN_SIZE).to(cfg.DEVICE)
    ppo    = PPO(agent, cfg)
    buffer = RolloutBuffer(cfg.STEPS_PER_ROLLOUT, obs_dim, act_dim, cfg.DEVICE)

    obs, _      = env.reset()
    ep_ret      = 0.0
    ep_rets     = []
    num_updates = cfg.TOTAL_STEPS // cfg.STEPS_PER_ROLLOUT

    for update in range(1, num_updates + 1):
        buffer.clear()
        for step in range(cfg.STEPS_PER_ROLLOUT):
            obs_t = torch.tensor(obs, dtype=torch.float32).to(cfg.DEVICE)

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_t)

            action_np = action.cpu().numpy().clip(act_low, act_high)
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.store(
                obs_t,
                action,
                log_prob,
                value.squeeze(),
                reward,
                done,
            )

            ep_ret += reward
            obs = next_obs

            if done:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            next_obs_t = torch.tensor(obs, dtype=torch.float32).to(cfg.DEVICE)
            next_value = 0.0 if buffer.done_buf[-1] else agent.get_value(next_obs_t).item()

        actor_loss, critic_loss, entropy = ppo.update(buffer, next_value)

        avg10 = np.mean(ep_rets[-10:]) if ep_rets else 0.0
        print(
            f"update {update:3d} | step {update * cfg.STEPS_PER_ROLLOUT:7d} | "
            f"avg_reward(10ep): {avg10:8.1f} | "
            f"actor_loss: {actor_loss:7.4f} | "
            f"critic_loss: {critic_loss:7.3f} | "
            f"entropy: {entropy:.3f}"
        )

    env.close()
    torch.save(agent.state_dict(), "ppo_pendulum.pth")
    print("Done. Model saved to ppo_pendulum.pth")


if __name__ == "__main__":
    train()
