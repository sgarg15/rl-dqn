# RL from Scratch

Implementing reinforcement learning algorithms from scratch in PyTorch, trained on Gymnasium environments.

---

## Phase 1: DQN (Deep Q-Network)

### Implementation

| File | Description |
|---|---|
| `dqn.py` | Neural network: state → Q-values for each action (2-layer MLP, 128 hidden units) |
| `replay_buffer.py` | Fixed-capacity deque storing `(state, action, reward, next_state, done)` tuples |
| `agent.py` | `DQNAgent` wrapping policy net, target net, optimizer, and replay buffer |
| `train.py` | Training loop with epsilon-greedy exploration and checkpoint saving |
| `eval.py` | Evaluation loop with optional live disturbances via arrow keys |
| `disturbance_wrapper.py` | Gymnasium wrapper that nudges the environment on keypress |

### How it works

DQN approximates the Q-function — the expected cumulative reward for taking action `a` in state `s`. Rather than learning a direct action policy, the agent learns Q-values for every action and always picks the highest one (with some exploration noise).

**Replay buffer** — instead of training on consecutive steps (which are highly correlated), transitions are stored in a buffer and sampled randomly. This breaks temporal correlation and makes gradient updates more stable.

**Target network** — a frozen copy of the policy network used to compute Bellman targets. Without it, the targets shift every step alongside the weights being updated a moving target problem that causes divergence. The target net is synced to the policy net every `target_update_freq` steps.

**Epsilon-greedy exploration** — with probability `ε` the agent picks a random action (explore); otherwise it picks `argmax Q(s, a)` (exploit). `ε` decays linearly from 1.0 to 0.05 over training so the agent explores early and exploits later.

**Bellman update:**
```
Q(s, a) = r + γ · max_a' Q_target(s', a') · (1 - done)
```
The `(1 - done)` term zeros out future rewards when the episode terminates, so the agent isn't rewarded for imagined futures after failure.

### Environments

**CartPole-v1** — balance a pole on a cart. State: `[x, x_dot, θ, θ_dot]`. Terminates if pole angle > ±12° or cart position > ±2.4. Truncates at 500 steps (success).

**LunarLander-v3** — land a spacecraft between two flags. State: position, velocity, angle, angular velocity, leg contact. Reward is shaped: +100 for landing, −100 for crashing, −0.3 per frame of main engine firing.

### Training

Switch environments by changing `ENV_NAME` at the top of `train.py`:

```python
ENV_NAME = "CartPole-v1"     # fast, converges in ~300 episodes
ENV_NAME = "LunarLander-v3"  # harder, needs ~400-500 episodes
```

```
python train.py
```

Checkpoints are saved only when the 50-episode rolling average improves. Training curves (rewards + TD loss) are saved to `training_curve.png`.

### Evaluation

```bash
python eval.py --env CartPole-v1 --model dqn_cartpole.pt
python eval.py --env LunarLander-v3 --model dqn_lunarlander.pt

# With live disturbances (arrow keys nudge the environment in real time)
python eval.py --env LunarLander-v3 --model dqn_lunarlander.pt --disturbance
```

### Key learnings

**DQN learns Q-values, not actions directly.** The network outputs one Q-value per action; the policy is just `argmax`. This means the same network implicitly represents the full policy without needing a separate policy head.

**Replay buffer breaks temporal correlation.** Consecutive transitions share state (s → s' are almost identical), which makes gradient updates noisy and correlated. Random sampling from a large buffer produces more independent, stable batches.

**Policy net vs target net.** Using the policy net to compute both the current Q-values and the Bellman targets creates a feedback loop improving the network shifts the targets, which shifts the network again. The frozen target net breaks this cycle. Without it, training is unstable.

**Hyperparameters heavily affect convergence.** With the current setup, training too long causes performance to collapse. Two causes: (1) epsilon decays to its floor early and the agent stops exploring, so it can't recover from a bad patch; (2) MSE loss with no gradient clipping allows a single bad batch to cause catastrophic weight updates, visible as a sharp loss spike followed by reward collapse. Fixes: Huber loss, gradient clipping, larger replay buffer, slower epsilon decay.
