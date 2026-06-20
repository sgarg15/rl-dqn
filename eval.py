import argparse

import gymnasium as gym
import torch

from dqn import DQN


# Take args from command line for env name and model path


parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="CartPole-v1")
parser.add_argument("--model", type=str, default="dqn_cartpole.pt")
args = parser.parse_args()

env = gym.make(args.env, render_mode="human")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim)
policy_net.load_state_dict(torch.load(args.model))
policy_net.eval()

state, info = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        q_values = policy_net(state_tensor)

    action = q_values.argmax(dim=1).item()

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    state = next_state
    total_reward += reward

print("Evaluation reward:", total_reward)

env.close()