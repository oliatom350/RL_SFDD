import numpy as np

class AgentSARSASemiGradiente:
    def __init__(self, state_dim, action_dim, alpha=0.01, gamma=0.99, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.weights = np.zeros((action_dim, state_dim))

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        q_values = self.weights @ state
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, next_action):
        q_sa = self.weights[action] @ state
        q_next = self.weights[next_action] @ next_state
        td_target = reward + self.gamma * q_next
        td_error = td_target - q_sa
        self.weights[action] += self.alpha * td_error * state