import random
from collections import deque
import numpy as np


class ReplayBuffer:
    """
A simple replay buffer for storing and sampling experience tuples.
    Args:
    capacity (int): Maximum number of experience tuples to store in the buffer.
    Methods:
    push(state, action, reward, next_state, done): Adds an experience tuple to the
        buffer.
    sample(batch_size): Randomly samples a batch of experience tuples from the buffer.  
    __len__(): Returns the current number of experience tuples stored in the buffer.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)