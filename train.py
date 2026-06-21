import gymnasium as gym
import matplotlib.pyplot as plt

from agent import DQNAgent

# --- Config ---
# ENV_NAME        = "LunarLander-v3"   # "CartPole-v1" or "LunarLander-v3"
ENV_NAME        = "CartPole-v1"   # "CartPole-v1" or "LunarLander-v3"

CONFIGS = {
    "MeaningOfArgs": dict(
        model_path         = "Path to save the trained model (e.g., dqn_cartpole.pt)",
        num_episodes       = "Total number of training episodes (e.g., 500)",
        epsilon_decay      = "Number of episodes over which epsilon decays (e.g., 500)",
        target_update_freq = "Number of steps between target network updates (e.g., 500)",
        learning_rate      = "Learning rate for the optimizer (e.g., 1e-3), it controls how much the model updates its parameters in response to the estimated error each time the model weights are updated.",
        gamma              = "Discount factor for future rewards (e.g., 0.99)",
        batch_size         = "Batch size for training (e.g., 64)",
        replay_capacity    = "Maximum capacity of the replay buffer (e.g., 10,000)",
    ),
    "CartPole-v1": dict(
        model_path         = "checkpoints/dqn_cartpole.pt",
        num_episodes       = 500,
        epsilon_decay      = 400,
        target_update_freq = 500,
        learning_rate      = 1e-3,
        gamma              = 0.99,
        batch_size         = 64,
        replay_capacity    = 10_000,
        use_replay_buffer  = True,
    ),
    "LunarLander-v3": dict(
        model_path         = "checkpoints/dqn_lunarlander.pt",
        num_episodes       = 500,
        epsilon_decay      = 400,
        target_update_freq = 800,
        learning_rate      = 1e-3,
        gamma              = 0.99,
        batch_size         = 64,
        replay_capacity    = 10_000,
        use_replay_buffer  = True,
    ),
}

cfg = CONFIGS[ENV_NAME]

epsilon_start = 1.0
epsilon_end   = 0.05
# ---------------

env = gym.make(ENV_NAME)
state, info = env.reset()

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=cfg["learning_rate"],
    gamma=cfg["gamma"],
    batch_size=cfg["batch_size"],
    replay_buffer_capacity=cfg["replay_capacity"],
)

episode_rewards = []
losses          = []
best_avg50      = float("-inf")
global_step     = 0

for episode in range(cfg["num_episodes"]):

    state, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Compute epsilon for the current step using exponential decay
        epsilon = max(epsilon_end, epsilon_start - (episode / cfg["epsilon_decay"]))
        action  = agent.select_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        # Shaping reward for CartPole to encourage keeping the pole centered
        if ENV_NAME == "CartPole-v1":
            reward -= 0.5 * abs(next_state[0]) / 2.4
        done = terminated or truncated

        if cfg["use_replay_buffer"]:
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
        else:
            loss = agent.train_step_online(state, action, reward, next_state, done)

        if loss is not None:
            losses.append(loss)

        state = next_state
        episode_reward += reward
        global_step += 1

        if global_step % cfg["target_update_freq"] == 0:
            agent.update_target_network()

    episode_rewards.append(episode_reward)

    if episode % 10 == 0:
        avg10 = sum(episode_rewards[-10:]) / len(episode_rewards[-10:])
        avg50 = sum(episode_rewards[-50:]) / len(episode_rewards[-50:])

        buffer_str = f"Buffer: {len(agent.replay_buffer)}" if cfg["use_replay_buffer"] else "Buffer: off"
        print(
            f"Episode {episode}, ",
            f"Reward: {episode_reward:.1f}, ",
            f"Avg10: {avg10:.1f}, ",
            f"Avg50: {avg50:.1f}, ",
            f"Epsilon: {epsilon:.2f}, ",
            buffer_str,
        )

        if avg50 > best_avg50:
            best_avg50 = avg50
            agent.save(cfg["model_path"])
            print(f"  -> Checkpoint saved (best Avg50: {best_avg50:.1f})")

env.close()
print(f"Training complete. Best Avg50: {best_avg50:.1f}")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(episode_rewards, alpha=0.4, label="Episode reward")
if len(episode_rewards) >= 10:
    rolling10 = [
        sum(episode_rewards[max(0, i-9):i+1]) / min(i+1, 10)
        for i in range(len(episode_rewards))
    ]
    rolling50 = [
        sum(episode_rewards[max(0, i-49):i+1]) / min(i+1, 50)
        for i in range(len(episode_rewards))
    ]
    ax1.plot(rolling10, label="Avg10")
    ax1.plot(rolling50, label="Avg50")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward")
ax1.set_title(f"{ENV_NAME} — Training Rewards")
ax1.legend()

ax2.plot(losses, alpha=0.5)
ax2.set_xlabel("Training step")
ax2.set_ylabel("Loss")
ax2.set_title("TD Loss")

plt.tight_layout()
plt.savefig(f"plots/training_curve_{ENV_NAME}_{best_avg50:.1f}.png")
plt.show()
