import random
from collections import namedtuple, deque

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen = self.buffer_size)
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size = 64):
        return random.sample(self.memory, k = self.batch_size)
    
    def __len__(self):
        return len(self.memory)