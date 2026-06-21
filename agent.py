import random
import torch 
import torch.nn.functional as F

from dqn import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
            self, 
            state_dim,
            action_dim,
            learning_rate=1e-3,
            gamma=0.99,
            batch_size=64,
            replay_buffer_capacity=10000,
            device='cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Using Adam optimizer for training the policy network
        # An Adam optimizer works by maintaining a moving average of the gradients and their squares, which helps to adapt the learning rate for each parameter. This can lead to faster convergence and better performance compared to standard stochastic gradient descent (SGD), especially in cases where the loss landscape is complex or has sparse gradients.
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)  # Explore: select a random action
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        return q_values.argmax(dim=1).item()  # Exploit: select the action with max Q-value
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        current_q = self.policy_net(states_tensor).gather(1, actions_tensor)

        with torch.no_grad():
            max_next_q = self.target_net(next_states_tensor).max(dim=1)[0].unsqueeze(1)
            target_q = rewards_tensor + (self.gamma * max_next_q * (1 - dones_tensor))

        loss = F.huber_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)  # Gradient clipping to prevent exploding gradients
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())