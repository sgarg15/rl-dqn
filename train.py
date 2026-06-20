import gymnasium as gym
import torch
import torch.nn.functional as F
from dqn import DQN
from replay_buffer import ReplayBuffer

import random

import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

# Reset the environment to get the initial state and info
state, info = env.reset()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# The network we update at every timestep
policy_net = DQN(state_dim, action_dim)

# The copy used to compute the target Q-values, updated every few steps
target_net = DQN(state_dim, action_dim)

# Used to update the network weights
optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

# How much we care about the future reward
gamma = 0.99

# How many to train on at once
batch_size = 64

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Explore: select a random action
    
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        q_values = policy_net(state_tensor)
    
    return q_values.argmax(dim=1).item()  # Exploit: select the action with max Q-value

def train_step():
    if len(replay_buffer) < batch_size:
        return  # Not enough samples to train

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
    dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
    
    # The gather function is used to select the Q-values corresponding to the actions taken
    current_q = policy_net(states_tensor).gather(1, actions_tensor)

    with torch.no_grad():
        max_next_q = target_net(next_states_tensor).max(dim=1, keepdim=True)[0]
        # Bellman equation: Q(s, a) = r + gamma * max(Q(s', a')) * (1 - done) target (the 1-done term ensures that if the episode ended, we don't add future rewards)
        target_q = rewards_tensor + (gamma * max_next_q * (1 - dones_tensor))

    loss = F.mse_loss(current_q, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

replay_buffer = ReplayBuffer(capacity=10_000)

num_episodes = 300

epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 200

target_update_freq = 500
global_step = 0

episode_rewards = []

for episode in range(num_episodes):

    state, info = env.reset()
    episode_reward = 0
    done = False

    epsilon = max(epsilon_end, epsilon_start - (episode / epsilon_decay))

    while not done:
        action = select_action(state, epsilon=0.1) 

        next_state, reward, terminated, truncated, info = env.step(action)

        # Terminated is triggered when the pole angle is more than +- 12 deg or the cart moves beyond +- 2.4 units from the center
        # Truncated triggered when the time limit is reached
        done = terminated or truncated

        replay_buffer.push(state, action, reward, next_state, done)

        train_step()

        state = next_state
        episode_reward += reward
        global_step += 1

        if global_step % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
    episode_rewards.append(episode_reward)
    if episode % 10 == 0:
        avg_reward = sum(episode_rewards[-10:]) / len(episode_rewards[-10:])
        print(
            f"Episode {episode}, ",
            f"Reward: {episode_reward}, ",
            f"Average Reward (last 10): {avg_reward:.2f}, ",
            f"Epsilon: {epsilon:.2f}, ",
            f"Replay Buffer Size: {len(replay_buffer)}",
        )

torch.save(policy_net.state_dict(), "dqn_cartpole.pt")
print("Model saved as dqn_cartpole.pt")

plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN CartPole Training Rewards")
plt.savefig("training_curve.png")
plt.show()

env.close()


print("Final Replay Buffer Size:", len(replay_buffer))