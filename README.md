# PPO for Pendulum-v1

A clean implementation of **Proximal Policy Optimization (PPO)** in PyTorch, trained on the `Pendulum-v1` continuous control environment from Gymnasium.

Built from scratch to understand every component of PPO — GAE, clipped surrogate objective, value function learning, and return normalization.

---

## Results

| Metric | Value |
|---|---|
| Random policy baseline | ~ -1400 to -1600 |
| After 200k steps | ~ -400 to -500 |
| After 500k steps | ~ -200 to -300 (near optimal) |

```
update  91 | step 186368 | avg_reward(10ep):  -590.0 | actor_loss: -0.174 | critic_loss: 0.411 | entropy: 0.969
update  93 | step 190464 | avg_reward(10ep):  -501.5 | actor_loss:  0.107 | critic_loss: 0.093 | entropy: 1.004
update  96 | step 196608 | avg_reward(10ep):  -441.5 | actor_loss: -0.169 | critic_loss: 0.094 | entropy: 1.090
```

---

## Environment

**Pendulum-v1** — a rigid rod attached to a fixed pivot. The agent applies torque to swing it upright and balance it.

- **Observation** `(3,)` — `[cos θ, sin θ, θ_dot]`
- **Action** `(1,)` — torque in `[-2, 2]`
- **Reward** — negative function of angle, angular velocity, and torque. Maximum is `0` (perfect balance), minimum is `-16.27` per step
- **Episode length** — 200 steps → optimal total reward ≈ `-200`

---

## Project Structure

```
ppo_pendulum_project/
├── config.py                 # All hyperparameters
├── train.py                  # Training loop
├── simulate.py               # Load model, record video
├── env/
│   └── make_env.py           # Gymnasium wrapper
├── models/
│   ├── actor.py              # Gaussian policy network
│   ├── critic.py             # Value network V(s)
│   └── actor_critic.py       # Combines actor + critic
├── memory/
│   └── rollout_buffer.py     # Pre-allocated rollout storage
├── algorithms/
│   └── ppo.py                # PPO update — clipped objective, GAE, grad clipping
├── utils/
│   └── gae.py                # Generalized Advantage Estimation + RunningMeanStd
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/your-username/ppo-pendulum
cd ppo-pendulum
pip install -r requirements.txt
```

Requirements: `torch`, `gymnasium`, `numpy`. For video recording also install `ffmpeg`:

```bash
pip install imageio[ffmpeg]
```

---

## Training

```bash
python train.py
```

Logs one line per update (every 2048 steps):

```
update  30 | step  61440 | avg_reward(10ep): -1246.8 | actor_loss: -0.299 | critic_loss: 6269.0 | entropy: 1.285
```

Model is saved to `ppo_pendulum.pth` after training.

---

## Simulation

```bash
python simulate.py
```

Loads `ppo_pendulum.pth` and runs 5 deterministic episodes (mean action, no noise). Videos are saved as `.mp4` files in `videos/`.

---

## Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `TOTAL_STEPS` | `200,000` | Extend to `500k` for full convergence |
| `STEPS_PER_ROLLOUT` | `2048` | Steps collected before each PPO update |
| `UPDATE_EPOCHS` | `10` | Passes over rollout data per update |
| `MINIBATCH_SIZE` | `64` | 2048 / 64 = 32 minibatches per epoch |
| `GAMMA` | `0.99` | Discount factor |
| `LAMBDA` | `0.95` | GAE lambda — bias/variance tradeoff |
| `CLIP_EPS` | `0.2` | PPO clip range |
| `LR` | `3e-4` | Adam learning rate |
| `MAX_GRAD_NORM` | `0.5` | Gradient clipping threshold |
| `HIDDEN_SIZE` | `64` | Hidden layer width |
| `VF_COEF` | `0.5` | Critic loss weight in total loss |

---

## Implementation Notes

### Generalized Advantage Estimation
Advantages are computed backwards through the rollout using the TD error at each step:

```
delta_t     = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
advantage_t = delta_t + gamma * lambda * (1 - done) * advantage_{t+1}
```

A bootstrap value `V(s_T)` is used for the final state when the rollout was truncated rather than truly terminated.

### Return Normalization
Raw Pendulum returns range from roughly `-1600` to `0`. Without normalization, early critic MSE losses reach `3000–8000`, and gradient clipping prevents the critic from learning fast enough. A `RunningMeanStd` tracker divides returns by their rolling standard deviation (not subtracting mean — sign must be preserved) before computing critic loss, keeping it in the `0.05–1.0` range throughout training.

### Orthogonal Initialization
All layers use orthogonal init. The actor output layer uses `gain=0.01` so the policy starts near-uniform across actions, ensuring exploration before exploitation.

### Clipped Surrogate Objective
```
ratio       = exp(log_prob_new - log_prob_old)
actor_loss  = -mean(min(ratio * A, clip(ratio, 1-eps, 1+eps) * A))
```

The clip removes the gradient whenever the policy moves outside the trust region `[1-ε, 1+ε]`, preventing destructively large updates.

---

## Reference

- Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Schulman et al. (2016) — [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — reference implementation
