import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
sys.path.append('src/')
from QNet import DQN_Network

class AgentDQLearning:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, epsilon=1.0, min_epsilon=0.01, decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

        self.q_net = DQN_Network(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)

        q_val = self.q_net(state_tensor)[action]
        with torch.no_grad():
            q_next = torch.max(self.q_net(next_state_tensor))
            target = reward + (0 if done else self.gamma * q_next)

        loss = self.criterion(q_val, torch.tensor(target))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)