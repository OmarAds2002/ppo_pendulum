import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from config              import Config
from models.actor_critic import ActorCritic


def simulate(model_path="ppo_pendulum.pth", episodes=5, video_folder="videos"):
    cfg = Config()

    env = gym.make(cfg.ENV_NAME, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda ep: True,
    )

    obs_dim  = env.observation_space.shape[0]
    act_dim  = env.action_space.shape[0]
    act_low  = env.action_space.low
    act_high = env.action_space.high

    agent = ActorCritic(obs_dim, act_dim, cfg.HIDDEN_SIZE).to(cfg.DEVICE)
    agent.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    agent.eval()

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        done   = False

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(cfg.DEVICE)
            with torch.no_grad():
                # get mean action directly — deterministic, no noise
                action = agent.actor.network(obs_t)

            action_np = action.cpu().numpy().clip(act_low, act_high)
            obs, reward, terminated, truncated, _ = env.step(action_np)
            ep_ret += reward
            done = terminated or truncated

        print(f"Episode {ep + 1}: reward = {ep_ret:.1f}")

    env.close()
    print(f"Videos saved to '{video_folder}/'")


if __name__ == "__main__":
    simulate()