import gymnasium as gym
import numpy as np

from agent import DQNAgent

# env = gym.make("CartPole-v1")
env = gym.make("LunarLander-v3")

# Reset the environment to get the initial state and info
state, info = env.reset()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=1e-3,
    gamma=0.99,
    batch_size=64,
    replay_buffer_capacity=10_000,
)

num_episodes = 500

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 200

target_update_freq = 500
global_step = 0

episode_rewards = []
losses = []

for episode in range(num_episodes):

    state, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        # Compute epsilon for the current step using exponential decay
        epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay))

        action = agent.select_action(state, epsilon)

        next_state, reward, terminated, truncated, info = env.step(action)

        # Terminated is triggered when the pole angle is more than +- 12 deg or the cart moves beyond +- 2.4 units from the center
        # Truncated triggered when the time limit is reached
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, done)

        loss = agent.train_step()

        if loss is not None:
            losses.append(loss)

        state = next_state
        episode_reward += reward
        global_step += 1

        if global_step % target_update_freq == 0:
            agent.update_target_network()

    episode_rewards.append(episode_reward)
    if episode % 10 == 0:
        avg_reward_10 = sum(episode_rewards[-10:]) / len(episode_rewards[-10:])
        avg_reward_50 = sum(episode_rewards[-50:]) / len(episode_rewards[-50:])
            
        print(
            f"Episode {episode}, ",
            f"Reward: {episode_reward}, ",
            f"Average Reward (last 10): {avg_reward_10:.2f}, ",
            f"Average Reward (last 50): {avg_reward_50:.2f}, ",
            f"Epsilon: {epsilon:.2f}, ",
            f"Replay Buffer Size: {len(agent.replay_buffer)}",
        )

if env.spec.id == "LunarLander-v3":
    file_name = "dqn_lunarlander.pt"
elif env.spec.id == "CartPole-v1":
    file_name = "dqn_cartpole.pt"
else :
    file_name = "dqn_model.pt"
    
agent.save(file_name)
print(f"Model saved as {file_name}")

env.close()


print("Final Replay Buffer Size:", len(agent.replay_buffer))