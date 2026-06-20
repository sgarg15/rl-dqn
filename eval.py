import argparse

import gymnasium as gym

from agent import DQNAgent
from disturbance_wrapper import DisturbanceWrapper


parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="CartPole-v1")
parser.add_argument("--model", type=str, default="dqn_cartpole.pt")
parser.add_argument("--disturbance", action="store_true", help="Enable live disturbances via left/right arrow keys")
parser.add_argument("--nudge", type=float, default=0.5, help="Disturbance strength (default: 0.5)")
parser.add_argument("--impulse", action="store_true", help="Single impulse per keypress instead of hold")
args = parser.parse_args()

env = gym.make(args.env, render_mode="human")
if args.disturbance:
    env = DisturbanceWrapper(env, strength=args.nudge, impulse=args.impulse)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
agent.load(args.model)

episode = 0

try:
    while True:
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode += 1

        while not done:
            action = agent.select_action(state, epsilon=0.0)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode} reward: {total_reward:.1f}")

except KeyboardInterrupt:
    pass

finally:
    env.close()
